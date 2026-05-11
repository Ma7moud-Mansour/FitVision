from abc import ABC, abstractmethod


class ExerciseEvaluator(ABC):
    def __init__(self):
        self.counter = 0        # عداد العدات
        self.stage = None       # الحالة (نازل ولا طالع - Down/Up)
        self.feedback = ""      # رسالة لليوزر

    @abstractmethod
    def evaluate(self, landmarks, w, h, engine):
        """اللوجيك الخاص بكل تمرينة هيتكتب هنا"""
        pass

    def get_ml_prediction(self, landmarks):
        """
        Override point for ML-enabled evaluators.
        Returns (label, confidence) or None if ML is not available.
        """
        return None