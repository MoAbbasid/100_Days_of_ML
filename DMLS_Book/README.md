# 📘 Designing Machine Learning Systems by Chip Huyen — Chapter 1 Notes

This chapter introduces what it really takes to bring machine learning (ML) into real-world production environments. It sets the stage for the rest of the book by highlighting the differences between academic ML and production ML, and the kinds of systems thinking required to deploy ML successfully at scale.

---

## 🌐 Links

[![Chip Huyen GitHub](https://img.shields.io/badge/Chip%20Huyen-GitHub-white)](https://github.com/chiphuyen/dmls-book)  
[![MLOps Discord](https://img.shields.io/badge/Join-MLOps%20Discord-blue)](https://discord.com/invite/Mw77HPrgjF)

---

## 🧠 What Is Machine Learning?

At its core, machine learning is a way to:
1. **Learn complex patterns** from
2. **Existing data** and
3. **Use those patterns to make predictions** on
4. **New, unseen data**

For supervised learning specifically, models learn by using example **input-output pairs**. For instance, predicting Airbnb prices based on listing characteristics like size, location, and amenities.

> 💡 Unlike relational databases (which require explicit relationships between data columns), ML systems learn relationships from data patterns.

---

## ✅ When Is ML a Good Fit?

ML isn't a universal solution. It's best suited for:
- Problems that involve **repetitive patterns**
- Tasks where **lots of data is available**
- Scenarios where predictions can **afford to be wrong** (e.g., recommending a movie vs. driving a car)

Additionally:
- ML systems benefit from **scale** — high investment pays off when widely used.
- ML is often **one part of a larger solution**, not the whole thing.

---

## 🏢 ML in Production vs. ML in Research

| Aspect                | ML Research                        | ML Production                          |
|----------------------|------------------------------------|----------------------------------------|
| **Data**             | Clean, well-labeled, historical    | Noisy, unstructured, constantly changing |
| **Latency & Throughput** | Less critical                     | Critical for user experience & cost    |
| **Fairness/Bias**    | Often theoretical discussion       | Real-world impact at scale             |
| **Stakeholders**     | Mostly researchers                 | Cross-functional teams (PMs, sales, infra) |
| **Interpretability** | Optional                           | Sometimes required (e.g., healthcare, finance) |

Production ML also involves thinking beyond the model:
- Data pipelines
- Deployment
- Monitoring
- Retraining
- Cost-efficiency
- Organizational alignment

---

## ⚙️ ML Systems ≠ Traditional Software

Traditional software is mostly rule-based — you program logic directly.

ML systems:
- Learn patterns from data
- Require **model training and evaluation**
- Need continuous **monitoring and updates**
- Can produce **unintended bias** due to historical data

---

## 🤝 Conflicting Stakeholder Goals: A Real Example

In a restaurant recommendation app (which earns revenue from a 10% service fee), different teams want different things:

- **ML Engineers**: Want the most accurate model
- **Sales Team**: Prefer recommending expensive restaurants to increase profit
- **Product Team**: Focused on low latency (e.g., under 100ms) to keep users from bouncing
- **ML Platform Team**: Concerned with system reliability and scaling, not frequent model changes
- **Manager**: Just wants higher margins — maybe even by cutting the ML team

This shows why ML development isn't just a technical challenge — it's also about aligning diverse priorities.

---

## ⚡ Batching: Latency vs. Throughput

- In single-query systems:
  - **Higher latency = lower throughput**
  - E.g., 10ms/query → 100 queries/sec; 100ms/query → 10 queries/sec

- In batched systems:
  - You can **increase both latency and throughput**
  - E.g., 50 queries in 20ms → 2,500 queries/sec
  - 🔑 Key insight: Batching leverages parallelism for efficiency, but it might slightly increase the time each user waits.

---

## ⚠️ Real-World Challenges

- ML models **encode the past**, not the future — so biases in data get amplified
- In production, you're not just dealing with model performance, but also:
  - Messy, shifting data
  - Latency budgets
  - System failures
  - Ethical considerations at scale

---

## 🌉 Takeaway

ML systems are **systems**, not just models. Success in production requires a holistic approach:
- Understand the use case
- Choose the right problem for ML
- Align across teams
- Design for reliability, scale, and ethics

This chapter reminds us that ML isn’t just about clever algorithms — it’s about thoughtful, systemic design choices.

---

📚 *These notes are based on Chapter 1 of* Designing Machine Learning Systems *by Chip Huyen.*
