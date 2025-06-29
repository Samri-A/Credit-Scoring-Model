## Credit Scoring Business Understanding

### 1. Why Basel II Pushes Us Toward Interpretable, Well‑Documented Models  
Basel II requires banks to measure credit risk accurately and to explain that measurement to regulators and auditors. A model that is easy to understand—showing how each factor (age, income, past payment behaviour, etc.) affects risk—reduces audit time, speeds up regulatory approval, and builds trust with senior management. Clear documentation also makes future maintenance easier when regulations or business priorities change.

---

### 2. Why We Build a “Proxy” Default Variable—and the Risks Involved  
Our dataset does not carry an explicit “default = Yes/No” label. To train any supervised model, we therefore create a **proxy**—for example, classifying a customer who is 90 days past due as a “default.”  
*Benefits*:  
* Allows us to train and test the model with historical data.  
* Enables early identification of at‑risk accounts before a legal default occurs.  

*Risks*:  
* If the proxy is too strict or too lenient, the model may learn patterns that do **not** reflect real default behaviour.  
* Mis‑calibration can lead to lost revenue (rejecting good borrowers) or higher losses (accepting risky borrowers), and could attract regulatory criticism for unfair lending practices.

---

### 3. Choosing Between Simple and Complex Models in a Regulated Setting  

| Aspect | Simple, Interpretable Model (e.g., Logistic Regression + WoE) | Complex, High‑Performance Model (e.g., Gradient Boosting) |
|--------|--------------------------------------------------------------|-----------------------------------------------------------|
| **Transparency** | High—each variable’s impact is clear and can be explained in plain language. | Low—interactions are non‑linear and harder to justify. |
| **Regulatory Acceptance** | Usually straightforward; documentation and validation are well‑established. | Requires extra effort (e.g., model explainability tools) and may face more questions. |
| **Predictive Power** | Adequate but may miss subtle patterns. | Often higher accuracy and better capture of non‑linear relationships. |
| **Operational Risk** | Easier to monitor, back‑test, and recalibrate. | More complex to monitor; risk of unnoticed drift or overfitting. |
| **Implementation Cost** | Lower—both in development time and compute resources. | Higher—needs tuning, stronger governance, and potentially more powerful hardware. |

**Bottom line:** In highly regulated environments, a slight drop in predictive performance is often acceptable if it means the model is transparent, auditable, and easier to defend to regulators.

---
