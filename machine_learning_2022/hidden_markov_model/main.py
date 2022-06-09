from hidden_markov_model import HMM
import numpy as np


def main():
    states = ['subject', 'verb', 'object', 'end']
    A = np.array([
        [.5, .5, 0, 0], [0, 0, .7, .3],
        [0, 0, .5, .5], [0, 0, 0, 1]]
    )

    observations = ['boat', 'man', 'old', 'rows', 'the', '.']
    B = np.array([
        [.2, 0, .2, 0], [.2, .2, .2, 0],
        [.2, 0, .2, 0], [.1, .8, .1, 0],
        [.3, 0, .3, 0], [0, 0, 0, 1]
    ])

    PI = np.array([1, 0, 0, 0])

    hmm = HMM(states, observations, A, B, PI)

    hmm.show()
    prob, prob_path = hmm.viterbi_algorithm(['the', 'man', 'rows', '.'])
    print(f'#Viterbi\nThe probability: {prob}')
    print(f'Best path: {prob_path}')


if __name__ == '__main__':
    main()
