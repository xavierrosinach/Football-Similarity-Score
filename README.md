# Football-Similarity-Score

This repository contains a sports analytics project designed to identify similar football players using advanced statistical data. The goal is to provide a **data-driven scouting tool** that enables objective comparison between players across different teams, leagues, and competitive contexts.

This project is intended to:

- Identify **potential replacements** for players  
- Support **scouting and recruitment decisions**  
- Analyze the transfer market using **quantitative methods**  
- Complement traditional scouting based on video and observation  

---

## Application Scope

The model has been applied to the **top teams worldwide according to the ELO ranking**, generating similarity reports for each player within these teams.

The full implementation and execution pipeline can be found in the project code available in this repository (link to be added).

For each team, the system automatically generates a dataset containing the most similar players for every squad member.

---

## Methodology

### Data Processing

The system processes match-level and player-level data to build a season-wide representation of each player.  
This includes data aggregation, feature engineering, and transformation into a structured numerical format suitable for modeling.

---

### Similarity Model

Player similarity is computed using **cosine similarity**.

Cosine similarity is a metric that measures the similarity between two vectors by calculating the cosine of the angle between them.  
Instead of focusing on absolute values, it evaluates how similar the **patterns and distributions of features** are between players.

- A value close to **1** indicates very similar players  
- A value close to **0** indicates low similarity  

This approach is particularly suitable for high-dimensional sports data, where the relative contribution of features is more important than their magnitude.

---

## Output

The model produces:

- A ranked list of the **most similar players** for each player  
- A **normalized similarity score** (0–100%)  
- Team-level outputs in `.csv` format  

---

## Use Cases

This system can be applied in:

- Professional **scouting departments**  
- **Recruitment and transfer analysis**  
- **Player benchmarking**  
- Research in sports analytics  

---

## Conclusion

This project demonstrates how advanced metrics and machine learning techniques can be used to build a robust **player similarity model**, enabling more informed and objective decision-making in modern football.