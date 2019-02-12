from overrides import overrides
 
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
 
@Predictor.register('smile')
class SmileDBPredictor(Predictor):
     
    def predict(self, df) -> JsonDict:
        return self.predict_batch_json(df)
     
    @overrides
    def _batch_json_to_instances(self, df):
        instances = []
        for i in df.index:
            instances.append(self._json_to_instance(df.loc[i]))
        return instances
 
    @overrides
    def _json_to_instance(self, row) -> Instance:
        return self._dataset_reader.text_to_instance(row[1])