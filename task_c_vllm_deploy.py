#!/usr/bin/env python3
"""
任务 C：vLLM 本地部署与性能测试
- 调用本地 vLLM 服务
- 对比 vLLM 本地部署 vs agicto 云端 API 的性能
"""

import os
import time
from openai import OpenAI
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 60)
print("  任务 C：vLLM 本地部署与性能测试")
print("=" * 60)

# ── 配置 ──
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B")
AGICTO_API_KEY = os.environ.get("AGICTO_API_KEY", "sk-a1WcdiJnqNmcIozK9Y2scCvuqChNdFV7Qm3HgLePMM6pJG57")
AGICTO_BASE_URL = "https://api.agicto.cn/v1"
AGICTO_MODEL = "qwen-plus"

vllm_client = OpenAI(api_key="vllm-local", base_url=VLLM_BASE_URL)
cloud_client = OpenAI(api_key=AGICTO_API_KEY, base_url=AGICTO_BASE_URL)


def test_single_request(client, model, prompt, label):
    """测试单次请求"""
    print(f"\n--- 测试 {label} ---")
    print(f"Prompt: {prompt[:80]}...")
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=512,
        )
        elapsed = time.time() - start
        content = response.choices[0].message.content
        print(f"耗时: {elapsed:.2f}s")
        print(f"输出长度: {len(content)} 字符")
        print(f"回答（前200字）: {content[:200]}...")
        return elapsed, content
    except Exception as e:
        print(f"请求失败: {e}")
        return None, None


def benchmark_api(client, model, prompt, label, n_trials=3):
    """测试 API 的响应速度和输出质量"""
    times = []
    outputs = []

    for i in range(n_trials):
        print(f"  {label} Trial {i + 1}...")
        elapsed, content = test_single_request(client, model, prompt, label)
        if elapsed is not None:
            times.append(elapsed)
            outputs.append(content)

    if not times:
        return None

    avg_time = sum(times) / len(times)
    output_len = sum(len(o) for o in outputs) / len(outputs)

    return {
        "label": label,
        "avg_time": avg_time,
        "avg_output_len": output_len,
        "sample": outputs[0][:200] if outputs else "",
    }


def main():
    # ── 步骤 1：测试 vLLM 本地服务 ──
    print("\n--- 步骤1：测试 vLLM 本地服务 ---")
    test_single_request(
        vllm_client,
        VLLM_MODEL,
        "你好，请介绍一下番茄种植的基本要点。",
        "vLLM 本地服务",
    )

    # ── 步骤 2：性能对比 ──
    print("\n--- 步骤2：性能对比测试 ---")
    test_prompt = (
        "请详细介绍番茄早疫病的发病原因、症状识别和防治方法，"
        "包括农业防治和化学防治的具体措施。"
    )

    print("\n正在测试 vLLM 本地部署...")
    vllm_result = benchmark_api(vllm_client, VLLM_MODEL, test_prompt, "vLLM 本地")

    print("\n正在测试 agicto 云端 API...")
    cloud_result = benchmark_api(
        cloud_client, AGICTO_MODEL, test_prompt, "agicto 云端"
    )

    if vllm_result and cloud_result:
        # ── 步骤 3：可视化 ──
        print("\n--- 步骤3：生成对比图 ---")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        labels = [vllm_result["label"], cloud_result["label"]]
        times = [vllm_result["avg_time"], cloud_result["avg_time"]]
        colors = ["#4e79a7", "#e15759"]
        bars = ax1.bar(labels, times, color=colors, edgecolor="black")
        for bar, t in zip(bars, times):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                t + 0.1,
                f"{t:.2f}s",
                ha="center",
                fontsize=12,
                fontweight="bold",
            )
        ax1.set_ylabel("平均响应时间 (s)")
        ax1.set_title("响应速度对比")
        ax1.grid(True, alpha=0.3, axis="y")

        output_lens = [vllm_result["avg_output_len"], cloud_result["avg_output_len"]]
        bars = ax2.bar(labels, output_lens, color=colors, edgecolor="black")
        for bar, l in zip(bars, output_lens):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                l + 10,
                f"{l:.0f} 字符",
                ha="center",
                fontsize=12,
                fontweight="bold",
            )
        ax2.set_ylabel("平均输出长度 (字符)")
        ax2.set_title("回答详细程度对比")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig("task_c_performance_comparison.png", dpi=150, bbox_inches="tight")
        print("对比图已保存到 task_c_performance_comparison.png")

        print(f"\nvLLM 样例回答（前200字）:\n{vllm_result['sample']}...")
        print(f"\nagicto 样例回答（前200字）:\n{cloud_result['sample']}...")

    print("\n--- 任务 C 完成！---")
    print("思考题:")
    print("1. vLLM 相比直接用 HuggingFace 推理有什么优势？")
    print("2. 如果显存不够，有哪些方法可以降低显存占用？")
    print("3. 本地部署和云端 API 各有什么优劣？")


if __name__ == "__main__":
    main()
