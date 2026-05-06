def test_public_tasks_import():
    from pnpl.tasks import (
        PhonemeClassification,
        SpeechDetection,
        TaskProtocol,
        WordClassification,
        WordDetection,
    )

    assert TaskProtocol is not None
    assert SpeechDetection is not None
    assert PhonemeClassification is not None
    assert WordClassification is not None
    # Backwards-compat alias for code written against pnpl <= 0.1.0.
    assert WordDetection is WordClassification


def test_public_preprocessing_import():
    from pnpl.preprocessing import (
        BadChannels,
        BandpassFilter,
        Downsample,
        Epoch,
        HeadPosition,
        MaxwellFilter,
        NotchFilter,
        Pipeline,
        epochs_to_h5,
        fif_to_h5,
    )

    assert Pipeline is not None
    assert BadChannels is not None
    assert HeadPosition is not None
    assert MaxwellFilter is not None
    assert NotchFilter is not None
    assert BandpassFilter is not None
    assert Downsample is not None
    assert Epoch is not None
    assert fif_to_h5 is not None
    assert epochs_to_h5 is not None
