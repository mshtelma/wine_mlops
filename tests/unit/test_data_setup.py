from wine_classifier.dataset import WineDataset


def test_data_setup():
    df = WineDataset(11).prepare_features()
    df = df.rename(columns={col:col.replace(" ", "_") for col in df.columns})
    print(df.columns)