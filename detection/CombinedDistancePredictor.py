from EuclideanDistancePredictor import  *
from scipy.stats.stats import pearsonr
import operator

class CombinedDistancePredictor(EuclideanDistancePredictor):
    def __init__(self, dataset_path=None, folder_list=None, feature_file_postfix=None, window = 10, weights=[44, 24, 33]):
        EuclideanDistancePredictor.__init__(self, dataset_path, folder_list, feature_file_postfix)
        self.window = window
        self.weights = weights

    def prediction_window(self, ref, test):
        pear = {}
        corr = {}
        dist = {}
        for i in range(len(test)):
            p = {}
            c = {}
            d = {}
            for j in range(0,len(ref)):
                p[j] = abs(pearsonr(ref[j], test[i])[0])
                c[j] = np.correlate(ref[i], test[j])
                d[j] = np.linalg.norm(ref[i] - test[j])
            pear[i] = sorted(p.items(), key=operator.itemgetter(1), reverse = True)
            corr[i] = sorted(c.items(), key=operator.itemgetter(1), reverse = True)
            dist[i] = sorted(d.items(), key=operator.itemgetter(1))
        return [pear, corr, dist]

    def prediction_using_combined_distance_window(self,pear, corr, dist):
        pred = []
        for i in range(len(pear)):
            res = np.zeros(len(pear))
            p = pear[i]
            c = corr[i]
            d = dist[i]
            weight = self.window
            for j in range(0,self.window):
                (pi, v) = p[j]
                (ci, v) = c[j]
                (di, v) = d[j]
                res[pi] += weight * self.weights[0]
                res[ci] += weight * self.weights[1]
                res[di] += weight * self.weights[2]
                weight -= 1
            pred.append(np.argmax(res))
        return pred

    def prediction(self, ref, test):
        [pear, corr, dist] = self.prediction_window(ref, test)
        return self.prediction_using_combined_distance_window(pear, corr, dist)