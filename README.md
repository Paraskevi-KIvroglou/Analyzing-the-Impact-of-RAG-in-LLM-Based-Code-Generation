# Analyzing the Impact of RAG in LLM-Based Code Generation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Benchmark](https://img.shields.io/badge/Benchmark-MBPP%2B-orange)](https://github.com/evalplus/evalplus)

> **Master's Thesis Research** | IU University of Applied Sciences | M.Sc. Artificial Intelligence

This repository contains the complete implementation and experimental results from my Master's thesis investigating how **Retrieval Augmented Generation (RAG)** and **self-evaluation techniques** enhance Large Language Model performance in code generation tasks [file:1].

**Authors:** [Paraskevi Kivroglou]¹, [Prof. Dr. Martin, Simon]¹, [Prof. Dr-Ing. Schlippe, Tim,]¹

**Affiliations:** ¹IU Applied Sciences  

**Paper Type:** Master's Thesis

## 🎯 Key Findings

- **10.11% improvement** over baseline using optimized RAG with Qwen 2.5-Coder on MBPP+ benchmark [file:1]
- RAG combined with specific prompts and self-evaluation achieves **92.1% Pass@1** on 3K documents [file:1]
- Scaling knowledge bases from 1K to 3K documents progressively enhances accuracy [file:1]
- Self-evaluation alone reduces performance, but becomes powerful when combined with RAG [file:1]

## 📊 Performance Results

### Overall Performance Across Document Counts

![Performance by Document Count](images/pass-1_performance_document_count.jpg)

*Figure: Pass@1 performance comparison across all scenarios with varying vector store sizes (1K, 2K, 3K documents)* [attached_image:8]

### Detailed Results by Corpus Size

<table>
<tr>
<td><img src="images/pass-1_1000.jpg" alt="1K Documents Results" width="100%"/></td>
<td><img src="images/pass-1_2000.jpg" alt="2K Documents Results" width="100%"/></td>
</tr>
<tr>
<td><img src="images/pass-1_3000.jpg" alt="3K Documents Results" width="100%"/></td>
<td><img src="images/pass-1_performance_document_count.jpg" alt="Combined Results" width="100%"/></td>
</tr>
</table>

*Pass@1 scores across experimental scenarios for 1K, 2K, and 3K document corpus sizes* [attached_image:2][attached_image:11][attached_image:7][attached_image:8]

## 🔬 Experimental Scenarios

This research evaluates **7 distinct scenarios** to isolate the impact of RAG, self-evaluation, and specialized prompting [file:1]:

### Scenario 1a: Baseline

Standard Qwen 2.5-Coder without any enhancements [file:1]

<img src="images/scenario1.jpg" alt="Scenario 1a" width="400"/>

[attached_image:9]

---

### Scenario 1b: Self-Evaluation Only

Baseline with self-evaluation prompts (no RAG) [file:1]

<img src="images/Scenario1b.jpg" alt="Scenario 1b" width="500"/>

[attached_image:3]

---

### Scenario 2a: Basic RAG

Introduction of RAG with standard prompting [file:1]

<img src="images/scenario2a-new.jpg" alt="Scenario 2a" width="500"/>

[attached_image:10]

---

### Scenario 2b: Basic RAG + Self-Evaluation

RAG combined with self-evaluation for iterative refinement [file:1]

<img src="images/scenario2b-new.jpg" alt="Scenario 2b" width="500"/>

[attached_image:4]

---

### Scenario 3a: RAG + Specific Prompts

Optimized RAG with task-specific prompting strategies [file:1]

<img src="images/scenario3a.jpg" alt="Scenario 3a" width="500"/>

[attached_image:1]

---

### Scenario 3b: RAG + Specific Prompts + Self-Evaluation

**Best performing configuration** - combines all techniques [file:1]

<img src="images/scenario3b.jpg" alt="Scenario 3b" width="600"/>

[attached_image:5]

---

### Scenario 4: Enhanced RAG + Self-Evaluation + Code Execution

Adds code execution tool for runtime validation [file:1]

<img src="images/scenario4-new.jpg" alt="Scenario 4" width="700"/>

[attached_image:12]

---

## 📈 Key Performance Metrics

| Scenario | 1K Docs | 2K Docs | 3K Docs | vs Baseline |
|----------|---------|---------|---------|-------------|
| **1a: Baseline** | 83.6% | 83.6% | 83.6% | - |
| **1b: Self-Eval Only** | 77.2% | 77.2% | 77.2% | -6.4% |
| **2a: Basic RAG** | 82.5% | 84.1% | 86.8% | +3.2% |
| **2b: RAG + Self-Eval** | 84.9% | 85.7% | 89.0% | +5.4% |
| **3a: RAG + Specific Prompts** | 88.6% | 87.3% | 87.8% | +4.2% |
| **3b: RAG + Specific + Self-Eval** | 90.7% | 91.5% | **92.1%** | **+10.11%** ✅ |
| **4: Enhanced RAG** | 91.5% | 91.5% | 92.1% | +10.11% |

[file:1]

## 🛠️ Technical Stack

- **LLM**: Qwen 2.5-Coder 32B (via Together AI API) [file:1]
- **Benchmark**: MBPP+ (Mostly Basic Programming Problems Plus) [file:1]
- **Evaluation Metric**: Pass@1 [file:1]
- **Embedding Model**: Semantic vector embeddings for retrieval [file:1]
- **Vector Store**: 1K/2K/3K document corpus configurations [file:1]
- **RAG Framework**: Custom implementation with retrieval and generation components [file:1]

## 🔑 Key Insights

### RAG Benefits

- Basic RAG provides modest improvements (2.5-3.2%) over baseline [file:1]
- Performance scales with corpus size - larger knowledge bases yield better results [file:1]
- RAG is particularly effective for domain-specific programming tasks [file:1]

### Self-Evaluation Impact

- Self-evaluation alone **decreases** performance by 6.4% due to overcorrection [file:1]
- Combined with RAG, self-evaluation becomes highly effective [file:1]
- Best results achieved when self-evaluation works with retrieved context [file:1]

### Optimal Configuration

- **Scenario 3b** (RAG + Specific Prompts + Self-Evaluation) achieves highest scores [file:1]
- Targeted prompting helps LLM discriminate and integrate retrieved knowledge [file:1]
- Code execution tools (Scenario 4) provide minimal additional gains [file:1]

## 📚 Research Questions Addressed

1. **How does RAG impact code generation performance?** RAG provides up to 10.11% improvement when optimally configured [file:1]

2. **What role does self-evaluation play?** Self-evaluation is detrimental alone but powerful when combined with RAG for identifying edge cases and logical errors [file:1]

3. **How does knowledge base size affect results?** Larger corpora (3K docs) consistently outperform smaller ones, especially in RAG scenarios [file:1]

⭐ **Star this repository if you find it useful for your research!**

---

## 🔗 Related Resources

- [Qwen 2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder)
- [MBPP+ Benchmark](https://github.com/evalplus/evalplus)
- [Full Thesis Document](link-to-thesis-pdf)

## Contact and Collaboration

Provide contact information for follow-up research:

Email: [paraskevikivroglou@gmail.com]

Supervisor: [Prof. Dr. Martin, Simon] ([simon.martin@iu.org])




