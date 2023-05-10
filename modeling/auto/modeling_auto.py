""" Custom Auto Model class."""

from modeling.longformer import LongformerModelForSequentialSentenceClassification


class AutoModelForSequentialSentenceClassification:

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return LongformerModelForSequentialSentenceClassification. \
            from_pretrained(pretrained_model_name_or_path,
                            config=kwargs['config'],
                            cache_dir=kwargs['cache_dir'],
                            revision=kwargs['revision'],
                            use_auth_token=kwargs['use_auth_token'],
                            )
