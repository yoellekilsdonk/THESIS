import pandas as pd

class analyze:
    def __init__(self, model):
        self.model = model
        self.predictions_pd = pd.concat([pd.DataFrame(model.dataset.ytestfold_finalfit[0]["id"].values),
                                         pd.DataFrame(model.dataset.ytestfold_finalfit[0]["hb"].values),
                                         pd.DataFrame(model.yhat)], axis=1)
        self.predictions_pd.columns = ["id", "hb", "yhat"]
        self.errors = self.predictions_pd["yhat"] - self.predictions_pd["hb"]

    def writecsv(self, path, suffix=''):
        path_csv = path + '/' + str(self.model.name) + suffix + '.csv'
        self.predictions_pd.to_csv(path_csv)
        return print("STEP 2.3 -- File '{}' has been written to your device.".format(path_csv))

    def perc_correctlyspec(self, threshold = 47):
        below47 = ((self.predictions_pd["hb"] <= threshold) & (self.predictions_pd["yhat"] <= threshold))
        above47 = ((self.predictions_pd["hb"] > threshold) & (self.predictions_pd["yhat"] > threshold))
        classification = ((below47 == True) | (above47 == True))
        return sum(classification) / len(classification)

    def perc_deviation(self, threshold = 5):
        return sum(self.errors.abs().values < threshold) / len(self.errors)


