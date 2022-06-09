import numpy as np
import schemdraw
from schemdraw import flow


class HMM:
    # todo add forward alg and train alg
    def __init__(
            self, states: list, observations: list,
            A: np.ndarray, B: np.ndarray, PI: np.ndarray) -> None:

        self.states = states
        self.observations = observations
        self.A = A  # to show states movement probabilities
        self.B = B  # to show observations probabilities
        self.PI = PI
        self.N = len(states)
        self.Obs = len(observations)
        self.margin = 3

    def show(self):
        with schemdraw.Drawing() as d:
            s = []
            for i, state in enumerate(self.states):
                s.append(flow.State().label(state).at((i*self.margin, 0)))
                d += s[i]

            for i in range(self.N):
                for j in range(self.N):
                    if self.A[i, j] == 0:
                        continue
                    elif i < j:
                        d += flow.Arc2(
                            arrow='->').at(s[i].SE).to(
                                s[j].SW).label(str(self.A[i, j]))
                    elif i > j:
                        d += flow.Arc2(
                            arrow='->').at(s[i].SW).to(
                                s[j].SE).label(str(self.A[i, j]))
                    else:
                        d += flow.ArcLoop(
                            arrow='->').at(s[i].SE).to(
                                s[j].SW).label(str(self.A[i, j]))

    def _b_s(self, state_idx, observation):
        i = self.observations.index(observation)
        return self.B[i, state_idx]

    @staticmethod
    def _v(viterbi, s2, t, a_s2_s, b_s):
        v = viterbi[s2, t-1] * a_s2_s * b_s
        return v

    def viterbi_algorithm(self, observations):
        T = len(observations)
        viterbi = np.zeros((self.N, T))
        backpointers = np.zeros((self.N, T))

        for s in range(self.N):
            viterbi[s, 0] = self.PI[s] * self._b_s(s, observations[0])

        for t in range(1, T):
            for s in range(self.N):
                v = []
                for s2 in range(self.N):
                    a_ij = self.A[s2, s]
                    b_s = self._b_s(s2, observations[t])
                    v.append(self._v(viterbi, s2, t, a_ij, b_s))
                viterbi[s, t] = max(v)
                backpointers[s, t] = np.argmax(v)

        best_path_prob = max(viterbi[:, T-1])

        best_path = []
        st = np.argmax(viterbi[:, T-1])
        best_path.append(st)
        for t in range(T-2, -1, -1):
            prev_st = np.argmax(viterbi[:, t])
            if self.A[prev_st, st] != 0:
                best_path.append(prev_st)
            else:
                prev_st += np.argmax(viterbi[prev_st+1:, t]) + 1
                best_path.append(prev_st)
            st = prev_st

        best_path = [self.states[x] for x in best_path]
        best_path.reverse()

        return best_path_prob, best_path
