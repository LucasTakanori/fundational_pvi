from src.packages import *

from sklearn.model_selection import train_test_split
from tqdm import tqdm

def set_deterministic(seed: int = 42): # for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_intervals_overlap(A: list[tuple[int,...]],
                              B: list[tuple[int,...]],
                              ) -> tuple[np.ndarray, list[tuple], list[tuple]]:
    '''
    1. This is equivalent to the adjacency matrix in graph theory.
    2. Here, we use full matrix because it is faster to lookup than sparse
    '''

    A_starts = np.array([a[0] for a in A])[:, np.newaxis]
    A_ends = np.array([a[-1] for a in A])[:, np.newaxis]
    B_starts = np.array([b[0] for b in B])
    B_ends = np.array([b[-1] for b in B])

    M = (A_starts < B_ends) & (B_starts < A_ends)

    if M.sum()>0:
        # sorting M, A, B based on conflicts (in descending order)
        rows = M.sum(axis=1).argsort()[::-1]
        cols = M.sum(axis=0).argsort()[::-1]

        M = M[np.ix_(rows, cols)]
        A = [A[idx] for idx in rows]
        B = [B[idx] for idx in cols]

    return M, A, B

class GraphBipartitePartitioner:
    """
    This class serves to split a sequential dataset into train and test INTERVALS without overlaps.
    The input is not the dataset itself, rather a list of 2-tuples representing the intervals to be extracted from the
    original dataset.

    ---
    Update 2025/10/11:
        - Change argument name 'active_mask' to 'intervals'
        - Remove arguments 'bounds'. Instead, allow 'intervals' to accept subgroups of intervals, and 'test_size' to accept
        list. The given subgroups are assumed disjoint. The test_size list gives local test_size for each subgroup.
    Update 2025/07/23:
        - Add an argument 'bounds' to localize block diagonal structure for large matrices.
        This is useful when we have a very long list of masks that span across multiple disjoint datasets.
        The matrices M and L will have a block diagonal structure.
        --> Complexity reduced from O((m1+m2+m3+...)*(n1+n2+n3+...)) to O(m1*n1 + m2*n2 + m3*n3 +...).
        (Is this equivalent to "stratified sampling" in statistics?)
    Update 2025/06/15:
        - Instead of recomputing and sorting the overlap matrices, just zero the rows and columns directly.
        - Make this a private subroutine for individual datasets --> delegated subroutine to be computed in parallel
    """
    def __init__(self,
                 intervals: list[tuple[int,...]],
                 test_size: list[float]|float=0.1,
                 random_state: int|np.random.RandomState=None,
                 shuffle: bool=True,
                 caller_id: str = None,
                 ) -> None:

        # need a way to access the caller class
        self._alias = type(self).__name__ if caller_id is None else caller_id

        intervals, test_size = self._validate_inputs(intervals, test_size)

        self.intervals = intervals
        self.test_size = test_size

        self.random_state = random_state

        self.test_size = test_size
        self.shuffle = shuffle

        self._skip: int = 1
        self._mod_base: int = 100 # how often to update history

        self.history = None

        # use this in test mode
        # self._orig_partition = self._split_sklearn()

    @staticmethod
    def _validate_inputs(intervals, test_size) -> tuple:
        # intervals can be:
        #   a list of tuples of ints, or
        #   a list of lists of tuples of ints.
        # In the first case, we have a list of masks. In the second cases, the masks were grouped into disjoint
        # portions, so we can speed up the operations.

        crit_a1 = isinstance(intervals, list) and isinstance(intervals[0], tuple) and isinstance(intervals[0][0], int)
        crit_a2 = isinstance(test_size, (float, int))
        crit_a3 = isinstance(test_size, (list, tuple)) and len(test_size) == 1

        crit_b1 = isinstance(intervals, list) and isinstance(intervals[0], list) and isinstance(intervals[0][0],
                                                                                                tuple) and isinstance(
                intervals[0][0][0], int)
        crit_b2 = isinstance(test_size, (float, int))
        crit_b3 = isinstance(test_size, (list, tuple)) and len(test_size) == len(intervals)

        if crit_a1:
            intervals = [intervals]
            if crit_a2:
                test_size = [test_size]
            elif crit_a3:
                pass
            else:
                raise RuntimeError("Mismatch type and length!")
        elif crit_b1:
            if crit_b2:
                test_size = [test_size] * len(intervals)
            elif crit_b3:
                pass
            else:
                raise RuntimeError("Mismatch type and length!")
        else:
            raise TypeError(f"Invalid type '{type(intervals)}' for intervals! Expect a list of tuples of ints, or a list of list of tuples of ints.")

        return intervals, test_size

    def _split_sklearn(self,
                       intervals: list[tuple[int,...]],
                       test_size: float=0.1,
                       ) -> tuple[list[tuple], list[tuple]]:

        """Naive get_partition from sklearn, does not consider overlaps"""
        if test_size==0.0:
            print(f"\t test_size={test_size} detected! Returning empty TEST masks!")
            return intervals, []

        elif test_size == 1.0:
            print(f"\t test_size={test_size} detected! Returning empty TRAIN masks!")
            return [], intervals

        else:
            A, B = train_test_split(intervals,
                                    test_size=test_size,
                                    random_state=self.random_state,
                                    shuffle=self.shuffle)

            return sorted(A), sorted(B)

    def split(self) -> tuple[list[tuple], list[tuple]]:
        groups = zip(self.intervals, self.test_size)

        A, B = [], []
        t1 = time.perf_counter()
        for k, (mg, ts) in enumerate(groups):
            print()
            print(f"{self._alias}: Splitting local subgroup {k+1}/{len(self.intervals)}...")
            Ak, Bk = self._bipartite_split(mg, ts)
            A.extend(Ak.copy())
            B.extend(Bk.copy())

        dt = time.perf_counter() - t1
        print()
        print(f"{self._alias}: Finish splitting all {len(self.intervals)} local subgroups. ({dt:.2f} seconds total)")

        A.sort()
        B.sort()
        print(f"\t Validating final split...")
        t2 = time.perf_counter()
        M = self.compute_overlap(A, B)[0]
        dt = time.perf_counter() - t2

        if M.sum() > 0:
            raise RuntimeError(f"Invalid partition! Found {int(M.sum())} overlaps!")
        else:
            fmt_ratio = lambda ratio: f"{round(ratio, 3)}|{round(1 - ratio, 3)}"
            rf = len(A) / (len(A) + len(B))

            print(f"\t Finish validation ({dt:.2f} seconds). No overlaps found.")
            print(f"Final partition: (Train|Test)=({len(A):,}|{len(B):,}), ratio=({fmt_ratio(rf)})")

            return A, B

    def _bipartite_split(self,
                         intervals: list[tuple[int,...]],
                         test_size: float=0.1,
                         ) -> tuple[list[tuple], list[tuple]]:

        if len(intervals) < 20:
            A, B = intervals, intervals.copy()
        else:
            A, B = self._split_sklearn(intervals, test_size) # call naive get_partition again

        if (not A) or (not B):
            return A, B

        t1 = time.time()

        M, A, B = self.compute_overlap(A, B)
        M0 = np.copy(M)
        Ak, Bk = A.copy(), B.copy()

        history = []
        history.append((M0, Ak, Bk))

        orig_ratio = len(A) / len(B)

        idxA_keep = set(range(len(A)))
        idxB_keep = set(range(len(B)))

        skip = self._skip

        pbar = tqdm(desc="\t Removing overlaps",
                    total=M.sum(),
                    bar_format='{l_bar}{bar:10}|{postfix}')

        for iteration in itertools.count():
            prev_overlaps = M.sum()
            if (len(Ak) <= skip) or (len(Bk) <= skip) or (M.sum()==0):
                break

            current_ratio = len(Ak) / len(Bk)
            if current_ratio >= orig_ratio:
                idxA_rm = M.sum(axis=1).argsort()[-skip:]
                M[idxA_rm, :] = False
                idxA_keep -= set(idxA_rm)
            else:
                idxB_rm = M.sum(axis=0).argsort()[-skip:]
                M[:, idxB_rm] = False
                idxB_keep -= set(idxB_rm)

            Ak = [A[idx] for idx in idxA_keep]
            Bk = [B[idx] for idx in idxB_keep]

            if (not (iteration+1) % (self._mod_base)) or M.sum()==0:
                history.append((np.copy(M), Ak, Bk))

            pbar.set_postfix_str(s=f"iter. #{iteration:,}: (Train|Test)=({len(Ak):,}|{len(Bk):,}), overlaps={M.sum():,}")
            pbar.update(prev_overlaps - M.sum())

        pbar.close()
        dt = time.time() - t1

        self.history = history

        rt = test_size
        ro = len(A)/(len(A)+len(B))
        rf = len(Ak)/(len(Ak)+len(Bk))

        fmt_ratio = lambda ratio: f"{round(ratio,3)}|{round(1-ratio,3)}"

        print(f"Finish splitting data masks. ({iteration+1} iterations in {dt:.2f} seconds)")
        print(f"Initial: (Train|Test)=({len(A):,}|{len(B):,}), ratio=({fmt_ratio(ro)}), overlaps={M0.sum()}")
        print(f"Final: (Train|Test)=({len(Ak):,}|{len(Bk):,}), ratio=({fmt_ratio(rf)}), overlaps={M.sum()}")
        print(f"Target: ratio=({fmt_ratio(rt)})")

        return Ak, Bk

    @staticmethod
    def compute_overlap(A: list[tuple[int,...]],
                        B: list[tuple[int,...]],
                        ) -> tuple[np.ndarray, list[tuple], list[tuple]]:

        #wrapper of function from outside
        M, A, B = compute_intervals_overlap(A,B)

        return M, A, B

if __name__ == "__main__":
    import random

    def random_partition(items: list, min_size: int, max_size: int) -> list:
        """
        Subdivide a list into sublists of random length between min_size and max_size
        """
        start = 0
        groups = []
        while start < len(items):
            remaining = len(items) - start
            if remaining < min_size:
                stop = len(items)
                groups[-1].extend(items[start:stop])
            else:
                stop = start + random.randint(min_size, min(max_size, remaining))
                groups.append(items[start:stop])

            start = stop

        return groups

    random.seed(42) # for native python random
    rng = np.random.RandomState(0) # for sklearn get_partition
    # rng = None # for sklearn get_partition

    N = 10_000 # Must be less than M
    M = 15_000

    bounds = [(0, 500), (505, 700), (705, 1000), (1005, 1400), (1500, 2000), (2050, 2300), (2400, 2700)]

    # bounds = [(0, 500), (400, 700), (705, 1000), (950, 1400), (1000, 2000), (2050, 2300), (2400, 2700)]

    mask_groups = []
    for b in bounds:
        idx_list = [random.randint(*b) for _ in range(500)]
        idx_list = sorted(set(idx_list))
        masks = [(a,a+5) for a in idx_list]
        mask_groups.append(masks)

    test_size = [round(random.uniform(0.,1.), 2) for _ in mask_groups]

    splitter = GraphBipartitePartitioner(intervals=mask_groups,
                                         test_size=0.2,
                                         random_state=rng)

    A2, B2 = splitter.split()
    M2 = splitter.compute_overlap(A2, B2)[0]
    print(f"\nRemaining overlaps: {M2.sum().item()}. (Expected 0)")