from pnpl.datasets import LibriBrainSpeech, LibriBrainPhoneme


def setup_dataset():
    speech_example_data = LibriBrainSpeech(
      data_path="./data/",
      include_run_keys = [("0","1","Sherlock1","1")],
      tmin=0.0,
      tmax=0.8,
    )
    sample_data, label = speech_example_data[0]

    # Print out some basic info about the sample
    print("Speech/Non-Speech:", sample_data.shape)
    print("Sample data shape:", sample_data.shape)
    print("Label shape:", label.shape)
    print("\n")

    phoneme_example_data = LibriBrainPhoneme(
      data_path="./data/",
      include_run_keys = [("0","1","Sherlock1","1"), ("0","1","Sherlock2","1"), ("0","1","Sherlock3","1")],
      tmin=0.0,
      tmax=0.8,
    )
    sample_data, label = phoneme_example_data[0]

    # Print out some basic info about the sample
    print("Phoneme Classification:", sample_data.shape)
    print("Sample data shape:", sample_data.shape)
    print("Label shape:", label.shape)

if __name__ == '__main__':
    setup_dataset()
