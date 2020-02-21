from src.landscape import Peak, PeakCollection
import numpy as np
import pandas as pd


def test_Peak_peak_coverage_works_hamming():
    test_peak = Peak(
        center_seq='AAAAAAA',
        seqs=['AAAAAAA', 'AAAABAA', 'AAABAAA', 'BAAAAAA', 'BBAABAA', 'BBBBBBB'],
        name='test peak',
        radius=3,
        dist_type='hamming',
        letterbook_size=2
    )
    assert test_peak.center_seq == 'AAAAAAA'
    np.testing.assert_array_almost_equal(
        test_peak.peak_coverage(max_radius=2).values[:, :-1],
        np.array([[1, 1],
                  [7, 3],
                  [21, 0]])
    )


def test_Peak_peak_coverage_works_edit():
    test_peak = Peak(
        center_seq='AAAAAAA',
        seqs=['AAAAAAA', 'AAAAAA', 'AAABAAA', 'BAAAAAA', 'BBAABAA', 'BBBBBBB'],
        name='test peak',
        radius=3,
        dist_type='edit',
        letterbook_size=2
    )
    assert test_peak.center_seq == 'AAAAAAA'
    np.testing.assert_array_almost_equal(
        test_peak.peak_coverage(max_radius=2).values[:, :-1],
        np.array([[1, 1],
                  [7, 3],
                  [21, 0]])
    )


def test_Peak_peak_abun_works_edit():
    test_peak = Peak(
        center_seq='AAAAAAA',
        seqs=['AAAAAAA', 'AAAAAAB', 'AAABAAA', 'BAAAAAA', 'BBAABAA', 'BBBBBBB'],
        name='test peak',
        radius=3,
        dist_type='edit',
        letterbook_size=2
    )

    peak_abun, peak_uniq_seq = test_peak.peak_abun(
        max_radius=2,
        table=pd.DataFrame(
            index=['AAAAAAA', 'AAAAAAB', 'BBAABAA'],
            columns=['sample1', 'sample2'],
            data=[[5, 3],
                  [0, 2],
                  [1, 1]]
        ),
        use_relative=False
    )
    peak_abun_expected = pd.DataFrame(
        columns=['sample1', 'sample2'],
        index=[0, 1, 2],
        data=[[5, 3],
              [0, 2],
              [0, 0]]
    )
    peak_uniq_expected = pd.DataFrame(
        columns=['sample1', 'sample2'],
        index=[0, 1, 2],
        data=[[1, 1],
              [0, 1],
              [0, 0]]
    )
    pd.testing.assert_frame_equal(peak_abun, peak_abun_expected)
    pd.testing.assert_frame_equal(peak_uniq_seq, peak_uniq_expected)


def test_Peak_peak_abun_works_edit_relative():
    test_peak = Peak(
        center_seq='AAAAAAA',
        seqs=['AAAAAAA', 'AAAAAAB', 'AAABAAA', 'BAAAAAA', 'BBAABAA', 'BBBBBBB'],
        name='test peak',
        radius=3,
        dist_type='edit',
        letterbook_size=2
    )

    peak_abun, peak_uniq_seq = test_peak.peak_abun(
        max_radius=2,
        table=pd.DataFrame(
            index=['AAAAAAA', 'AAAAAAB', 'BBAABAA'],
            columns=['sample1', 'sample2'],
            data=[[5, 3],
                  [0, 2],
                  [1, 1]]
        ),
        use_relative=True
    )
    peak_abun_expected = pd.DataFrame(
        columns=['sample1', 'sample2'],
        index=[0, 1, 2],
        data=[[1, 1],
              [0, 0.66666666666666],
              [0, 0]]
    )
    peak_uniq_expected = pd.DataFrame(
        columns=['sample1', 'sample2'],
        index=[0, 1, 2],
        data=[[1, 1],
              [0, 1],
              [0, 0]]
    )
    pd.testing.assert_frame_equal(peak_abun, peak_abun_expected, check_dtype=False)
    pd.testing.assert_frame_equal(peak_uniq_seq, peak_uniq_expected)


def test_PeakCollection_coverage():
    seqs = ['AAAAAAA', 'AAAAAAB', 'AAABAAA', 'BAAAAAA', 'BBAABAA', 'BBBBBAA']
    test_peak_1 = Peak(
        center_seq='AAAAAAA',
        seqs=seqs,
        name='test peak 1',
        radius=3,
        dist_type='hamming',
        letterbook_size=2
    )

    test_peak_2 = Peak(
        center_seq='BBBBBBB',
        seqs=seqs,
        name='test peak 2',
        radius=3,
        dist_type='hamming',
        letterbook_size=2
    )

    test_peak = PeakCollection([test_peak_1, test_peak_2])

    np.testing.assert_array_almost_equal(
        test_peak.peak_coverage(max_radius=2).values[:, :-1],
        np.array([[2, 1],
                  [14, 3],
                  [42, 1]])
    )


def test_PeakCollection_abun():
    seqs = ['AAAAAAA', 'AAAAAAB', 'AAABAAA', 'BAAAAAA', 'BBAABAA', 'BBBBBAA']
    test_peak_1 = Peak(
        center_seq='AAAAAAA',
        seqs=seqs,
        name='test peak 1',
        radius=3,
        dist_type='hamming',
        letterbook_size=2
    )

    test_peak_2 = Peak(
        center_seq='BBBBBBB',
        seqs=seqs,
        name='test peak 2',
        radius=3,
        dist_type='hamming',
        letterbook_size=2
    )

    test_peak = PeakCollection([test_peak_1, test_peak_2])

    peak_abun, peak_uniq_seq = test_peak.peak_abun(
        max_radius=2,
        table=pd.DataFrame(
            index=['AAAAAAA', 'AAAAAAB', 'BBAABAA', 'BBBBBAA'],
            columns=['sample1', 'sample2'],
            data=[[5, 3],
                  [0, 2],
                  [1, 1],
                  [1, 1]]
        ),
        use_relative=True
    )

    peak_abun_expected = pd.DataFrame(
        columns=['sample1', 'sample2'],
        index=[0, 1, 2],
        data=[[1, 1],
              [0, 0.66666666666666],
              [0.2, 0.333333333333]]
    )
    pd.testing.assert_frame_equal(peak_abun, peak_abun_expected, check_dtype=False)


def test_Peak_peak_abun_accept_seqs_not_defined():
    test_peak = Peak(
        center_seq='AAAAAAA',
        seqs=['AAAAAAA'],
        name='test peak',
        radius=3,
        dist_type='edit',
        letterbook_size=2
    )

    peak_abun, peak_uniq_seq = test_peak.peak_abun(
        max_radius=2,
        table=pd.DataFrame(
            index=['AAAAAAA', 'AAAAAAB', 'BBAABAA'],
            columns=['sample1', 'sample2'],
            data=[[5, 3],
                  [0, 2],
                  [1, 1]]
        ),
        use_relative=True
    )
    peak_abun_expected = pd.DataFrame(
        columns=['sample1', 'sample2'],
        index=[0, 1, 2],
        data=[[1, 1],
              [0, 0.66666666666666],
              [0, 0]]
    )
    peak_uniq_expected = pd.DataFrame(
        columns=['sample1', 'sample2'],
        index=[0, 1, 2],
        data=[[1, 1],
              [0, 1],
              [0, 0]]
    )
    pd.testing.assert_frame_equal(peak_abun, peak_abun_expected, check_dtype=False)
    pd.testing.assert_frame_equal(peak_uniq_seq, peak_uniq_expected)