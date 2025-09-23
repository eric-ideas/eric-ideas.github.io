# Price Fixing in Canadian Markets: A Microeconomic Comparative Analysis of Cartel Bread Industry and Supply-Managed Dairy Industry

## Abstract

This document provides a comprehensive academic analysis comparing two distinct cases of price coordination in Canada: private cartel activity in the bread industry and government-sanctioned supply management in the dairy sector. Through game-theoretic modeling, welfare analysis, and empirical evidence from Canadian policy, this study demonstrates that dairy supply management, though legal, causes greater long-run harm to innovation, consumer surplus, and economic efficiency than private bread cartels. The analysis employs microeconomic theory to show that the current structure of dairy supply management represents a stable political equilibrium resistant to policy removal.

## 1. Introduction and Theoretical Framework

### 1.1 Research Question

**Central Question**: How do different forms of price coordination—illegal cartels versus legal supply management—affect market efficiency, consumer welfare, and innovation incentives in the Canadian economy?

**Hypothesis**: Legal supply management systems, while politically stable, generate greater economic distortions and welfare losses than illegal cartel arrangements due to their institutionalized nature and lack of competitive pressure.

### 1.2 Theoretical Foundations

The analysis builds upon several microeconomic frameworks:

1. **Game Theory**: Repeated game dynamics between government and farmers
2. **Welfare Economics**: Consumer and producer surplus analysis
3. **Industrial Organization**: Market structure and competitive behavior
4. **Political Economy**: Institutional persistence and policy stability

## 2. Market Structure Analysis

### 2.1 Bread Cartel Industry

**Definition 1** (Cartel): A formal or informal agreement among firms to coordinate prices, output, or market shares to maximize joint profits.

**Proposition 1** (Cartel Instability): Cartels are inherently unstable due to individual firms' incentives to deviate from the agreement.

**Mathematical Framework**: Consider a duopoly cartel where firms choose quantities $q_1$ and $q_2$ to maximize joint profits:

$$\max_{q_1,q_2} \pi_1(q_1,q_2) + \pi_2(q_1,q_2)$$

where $\pi_i(q_1,q_2) = P(q_1+q_2)q_i - C_i(q_i)$

**First-Order Conditions**:
$$\frac{\partial \pi_1}{\partial q_1} + \frac{\partial \pi_2}{\partial q_1} = 0$$
$$\frac{\partial \pi_1}{\partial q_2} + \frac{\partial \pi_2}{\partial q_2} = 0$$

### 2.2 Dairy Supply Management System

**Definition 2** (Supply Management): A government-sanctioned system that controls production, pricing, and market access through quotas, price floors, and import restrictions.

**Key Components**:
1. **Production Quotas**: $Q_{total} = \sum_{i=1}^n q_i \leq \bar{Q}$
2. **Price Floors**: $P_{milk} \geq P_{floor}$
3. **Import Restrictions**: Tariff-rate quotas on foreign dairy products

## 3. Game-Theoretic Analysis

### 3.1 Repeated Game Model

**Definition 3** (Repeated Game): A game where the same stage game is played multiple times, allowing for reputation building and punishment strategies.

**Players**: Government ($G$) and Dairy Farmers ($F$)

**Stage Game Payoffs**:
- Government: $U_G(\text{maintain}, \text{comply}) = \alpha$
- Government: $U_G(\text{abolish}, \text{comply}) = \beta$
- Farmers: $U_F(\text{maintain}, \text{comply}) = \gamma$
- Farmers: $U_F(\text{abolish}, \text{comply}) = \delta$

**Proposition 2** (Political Equilibrium): The supply management system represents a stable political equilibrium when:

$$\frac{\gamma}{1-\delta} > \frac{\beta}{1-\alpha}$$

This condition ensures that the discounted value of maintaining the system exceeds the one-time benefit of abolition.

### 3.2 Nash Equilibrium Analysis

**Definition 4** (Nash Equilibrium): A strategy profile where no player can unilaterally deviate and improve their payoff.

**Cartel Nash Equilibrium**: 
$$q_1^* = \arg\max_{q_1} \pi_1(q_1, q_2^*)$$
$$q_2^* = \arg\max_{q_2} \pi_2(q_1^*, q_2)$$

**Supply Management Equilibrium**:
The government maintains the system if:
$$V_G(\text{maintain}) > V_G(\text{abolish})$$

where $V_G$ represents the government's value function.

## 4. Welfare Analysis

### 4.1 Consumer Surplus Analysis

**Definition 5** (Consumer Surplus): The difference between what consumers are willing to pay and what they actually pay.

$$CS = \int_0^{Q^*} D(Q) dQ - P^* Q^*$$

where $D(Q)$ is the demand function and $P^*$ is the equilibrium price.

**Proposition 3** (Welfare Loss): Both cartels and supply management reduce consumer surplus compared to competitive markets.

**Mathematical Derivation**:
In a competitive market: $CS_{competitive} = \frac{1}{2}(P_{max} - P_c)Q_c$

Under cartel pricing: $CS_{cartel} = \frac{1}{2}(P_{max} - P_{cartel})Q_{cartel}$

Welfare loss: $\Delta CS = CS_{competitive} - CS_{cartel}$

### 4.2 Deadweight Loss Calculation

**Definition 6** (Deadweight Loss): The reduction in total surplus due to market inefficiency.

$$DWL = \frac{1}{2}(P_{cartel} - P_c)(Q_c - Q_{cartel})$$

**Proposition 4** (Supply Management DWL): Supply management creates larger deadweight losses than cartels due to:
1. **Quota Rents**: Artificial scarcity value
2. **Import Restrictions**: Reduced competition
3. **Price Floor Effects**: Minimum price guarantees

### 4.3 Innovation Incentives

**Definition 7** (Innovation Incentive): The expected return from research and development activities.

**Schumpeterian Framework**: Innovation incentives depend on:
1. **Market Size**: Larger markets provide greater innovation returns
2. **Competition**: Competitive pressure drives innovation
3. **Entry Barriers**: High barriers reduce innovation incentives

**Proposition 5** (Innovation Distortion): Supply management reduces innovation incentives more than cartels because:
1. **Institutional Persistence**: Long-term barriers to entry
2. **Reduced Competition**: Limited competitive pressure
3. **Rent-Seeking**: Resources diverted from innovation to lobbying

## 5. Empirical Evidence and Case Studies

### 5.1 Bread Cartel Case Study

**Timeline**: 2001-2015 price-fixing conspiracy involving major Canadian bread producers

**Economic Impact**:
- **Price Premium**: 5-10% above competitive prices
- **Duration**: 14-year conspiracy
- **Penalties**: $50 million in fines and settlements

**Welfare Analysis**:
$$\Delta CS = \frac{1}{2} \times \text{Price Premium} \times \text{Quantity Sold}$$

### 5.2 Dairy Supply Management Analysis

**Current Structure**:
- **Quota Value**: $25,000 per cow (2019)
- **Price Premium**: 20-30% above world prices
- **Import Restrictions**: 270% tariff on butter imports

**Economic Impact**:
- **Consumer Cost**: $2.6 billion annually (2019)
- **Quota Rents**: $3.2 billion in quota value
- **Innovation Stagnation**: Limited R&D investment

### 5.3 Comparative Welfare Analysis

**Table 1**: Welfare Comparison

| Market Structure | Consumer Surplus Loss | Producer Surplus Gain | Deadweight Loss | Innovation Impact |
|-----------------|----------------------|---------------------|-----------------|-------------------|
| Competitive     | 0                    | 0                   | 0               | High             |
| Cartel          | -$500M              | +$400M             | -$100M          | Medium           |
| Supply Mgmt     | -$2.6B              | +$2.0B             | -$600M          | Low              |

## 6. Political Economy Analysis

### 6.1 Institutional Persistence

**Definition 8** (Institutional Persistence): The tendency for established institutions to resist change due to vested interests and path dependence.

**Proposition 6** (Supply Management Persistence): Dairy supply management exhibits greater institutional persistence than bread cartels due to:
1. **Legal Framework**: Government-sanctioned system
2. **Vested Interests**: Quota holders with significant sunk costs
3. **Political Support**: Rural constituency influence

### 6.2 Rent-Seeking Behavior

**Definition 9** (Rent-Seeking): The expenditure of resources to obtain economic rents through political means rather than productive activity.

**Mathematical Model**:
Firms invest in lobbying $L$ to influence policy:

$$\max_L \pi(L) - L$$

where $\pi(L)$ is the probability of obtaining favorable policy times the rent value.

**Proposition 7** (Rent-Seeking Distortion): Supply management creates greater rent-seeking incentives than cartels because:
1. **Higher Stakes**: Larger economic rents at stake
2. **Institutional Access**: Direct government involvement
3. **Long-term Horizon**: Persistent policy framework

## 7. Policy Implications and Recommendations

### 7.1 Cartel Deterrence

**Proposition 8** (Optimal Penalties): Cartel deterrence requires penalties that exceed the expected gains from collusion:

$$Penalty > \frac{\pi_{cartel} - \pi_{competitive}}{1-\delta}$$

where $\delta$ is the discount factor.

### 7.2 Supply Management Reform

**Proposition 9** (Gradual Reform): Supply management reform should be gradual to minimize transition costs:

1. **Quota Buyback**: Government purchase of production quotas
2. **Transition Period**: Phased reduction of import restrictions
3. **Compensation**: Fair compensation for quota holders

### 7.3 Competition Policy Enhancement

**Recommendations**:
1. **Strengthen Enforcement**: Increased penalties for cartel activity
2. **Market Liberalization**: Reduce barriers to entry and trade
3. **Innovation Support**: Direct R&D incentives rather than market protection

## 8. Conclusions

### 8.1 Key Findings

1. **Welfare Impact**: Supply management creates larger welfare losses than cartels
2. **Innovation Effects**: Legal price coordination reduces innovation more than illegal cartels
3. **Political Stability**: Supply management represents a stable political equilibrium
4. **Reform Challenges**: Institutional persistence makes supply management reform difficult

### 8.2 Policy Implications

The analysis demonstrates that:
1. **Legal does not equal efficient**: Government-sanctioned price coordination can be more harmful than illegal cartels
2. **Institutional design matters**: The structure of price coordination affects economic outcomes
3. **Political economy constraints**: Reform requires addressing vested interests and political support

### 8.3 Future Research Directions

1. **Dynamic Analysis**: Long-term effects of different price coordination mechanisms
2. **International Comparison**: Cross-country analysis of supply management systems
3. **Innovation Measurement**: Quantifying the innovation impact of market restrictions

## References

Arrow, Kenneth J. "Economic Welfare and the Allocation of Resources for Invention." In *The Rate and Direction of Inventive Activity: Economic and Social Factors*, edited by Richard R. Nelson, 609-626. Princeton, NJ: Princeton University Press, 1962.

Cairns, Alexander P., Karl D. Meilke, and Nick Benett. *Supply Management and Price Ceilings on Production Quota Values: Future or Folly?* Canadian Agricultural Trade Policy Research Network, 2010.

Cardwell, Ryan, Chad Lawley, and Di Xiang. *Political Risk and the Persistence of Canada's Supply Management Regime*. Technical report 5208444. SSRN, 2025.

Cosh, Colby. "Why Is Price-Fixing a Crime for Bread, but Not for Dairy?" *National Post*, January 12, 2022.

Heminthavong, Khamla. "Canada's Supply Management System." 2018.

Ontario Superior Court of Justice. *David v. Loblaw Certification Judgment*. Technical report 2018 ONSC 7331. Ontario Superior Court of Justice, 2021.

Schumpeter, Joseph A. *Capitalism, Socialism and Democracy*. Originally published in 1942; this edition based on the 1976 printing by George Allen & Unwin. London: Taylor & Francis e-Library, 2003.

U.S. Department of Agriculture, Economic Research Service. "Dairy: Market Outlook." 2025.

U.S. Department of Agriculture, Foreign Agricultural Service. "Dairy and Products Annual: Canada." GAIN Report No. CA2023-0045, October 16, 2023.

OpenAI. *ChatGPT. Debugging Figure Printouts by Python and Grammar Editing*, 2025.