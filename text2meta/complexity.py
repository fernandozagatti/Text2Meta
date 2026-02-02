import textstat
import re

class ComplexityFeatures:
    def __init__(self, data, text_column, target=None):
        self.data = data
        self.text_column = text_column
        self.target = target
        self.all_text = " ".join(data[self.text_column].astype(str).tolist())

    def flesch_reading_ease(self):
        return textstat.flesch_reading_ease(self.all_text)

    def smog_index(self):
        return textstat.smog_index(self.all_text)

    def flesch_kincaid_grade(self):
        return textstat.flesch_kincaid_grade(self.all_text)

    def coleman_liau_index(self):
        return textstat.coleman_liau_index(self.all_text)

    def automated_readability_index(self):
        return textstat.automated_readability_index(self.all_text)

    def dale_chall_readability_score(self):
        return textstat.dale_chall_readability_score(self.all_text)

    def difficult_words(self):
        return textstat.difficult_words(self.all_text)

    def linsear_write_formula(self):
        return textstat.linsear_write_formula(self.all_text)

    def gunning_fog(self):
        return textstat.gunning_fog(self.all_text)

    def school_grade(self):
        string_grade = textstat.text_standard(self.all_text, float_output=False)
        matches = re.findall(r"\d+", string_grade)
        school_grade = matches[0] if matches else 0
        return school_grade