from __future__ import annotations

import unittest

from backend.prompt_builder import PromptBuilder


class PromptBuilderTests(unittest.TestCase):
    def test_metadata_query_result_uses_match_reason_not_fake_similarity(self) -> None:
        builder = PromptBuilder()
        cells = [
            {
                "cell_id": "cell-1",
                "cell_type": "Hepatocyte",
                "metadata": {"tissue": "right lobe of liver"},
                "match_reasons": ["cell_type=Hepatocyte"],
            }
        ]

        context = builder.build_context(cells)

        self.assertIn("后端查询到的细胞数据", context)
        self.assertIn("Cell ID：cell-1", context)
        self.assertIn("命中：cell_type=Hepatocyte", context)
        self.assertNotIn("相似度 100.0%", context)

    def test_system_prompt_forbids_calling_backend_results_simulated(self) -> None:
        builder = PromptBuilder()

        messages = builder.build_messages(
            user_question="查询 cell_type 为 Hepatocyte 的细胞",
            retrieved_cells=[],
        )

        self.assertIn("不要声称自己是在模拟查询", messages[0]["content"])
        self.assertIn("后端真实查询结果", messages[0]["content"])
        self.assertIn("真实 Cell ID", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
