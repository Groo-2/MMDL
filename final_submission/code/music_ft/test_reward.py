import unittest

from reward import accuracy_reward, completion_text, score_one


class RewardTest(unittest.TestCase):
    def test_correct_letter(self):
        self.assertEqual(score_one("The clef is alto.\nAnswer: B", "B"), 1.0)

    def test_wrong_letter(self):
        self.assertEqual(score_one("...\nAnswer: C", "B"), 0.0)

    def test_markdown_and_option_text(self):
        self.assertEqual(score_one("**Answer:** B", "B"), 1.0)
        self.assertEqual(score_one("Answer: B. major second", "B"), 1.0)

    def test_no_answer_line_is_zero(self):
        # 잘린 응답: 'Answer:' 줄이 없다
        self.assertEqual(score_one("Wait... Perhaps the note is... Wait...", "A"), 0.0)

    def test_last_answer_line_wins(self):
        self.assertEqual(score_one("Answer: A\nHmm, rethinking.\nAnswer: D", "D"), 1.0)
        self.assertEqual(score_one("Answer: A\nHmm, rethinking.\nAnswer: D", "A"), 0.0)

    def test_out_of_range_letter(self):
        self.assertEqual(score_one("Answer: E", "A"), 0.0)

    def test_ambiguous_line_is_zero(self):
        self.assertEqual(score_one("Answer: A or B", "A"), 0.0)

    def test_empty(self):
        self.assertEqual(score_one("", "A"), 0.0)
        self.assertEqual(score_one([], "A"), 0.0)

    def test_conversational_formats(self):
        msg = [{"role": "assistant", "content": "Answer: C"}]
        parts = [{"role": "assistant", "content": [{"type": "text", "text": "Answer: C"}]}]
        self.assertEqual(completion_text(msg), "Answer: C")
        self.assertEqual(completion_text(parts), "Answer: C")
        self.assertEqual(accuracy_reward([msg, parts], ["C", "D"]), [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
