"""
Reward Functions
=================
Three types of reward signals:
  1. Rule-based (NLP metrics): coverage, precision, groundedness, redundancy
  2. Zero-shot LLM judge: uses the model itself to score quality
  3. Trained reward model: fine-tuned from SFT checkpoint on preference pairs

Combined reward = weighted sum of all active components.
"""

import re
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter

import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import nltk
    from nltk.translate.meteor_score import meteor_score as nltk_meteor
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

from config import PipelineConfig, RewardConfig

log = logging.getLogger(__name__)

# ================================================================
# 1. RULE-BASED REWARDS
# ================================================================

def parse_limitations(text: str) -> List[str]:
    """Extract individual limitation items from text."""
    if not text or not isinstance(text, str):
        return []

    lines = text.strip().split("\n")
    limitations = []

    for line in lines:
        line = line.strip()
        # Match bullet points, numbered items, or dashed items
        line = re.sub(r"^[\d]+[.)]\s*", "", line)
        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"^\*\*.*?\*\*:?\s*", "", line)  # Remove bold headers
        line = line.strip()
        if len(line.split()) >= 5:  # At least 5 words
            limitations.append(line)

    return limitations

def compute_coverage(generated: str, ground_truth: str) -> float:
    """
    Coverage: What fraction of ground truth limitations are captured?
    Uses TF-IDF cosine similarity for soft matching.
    """
    gt_lims = parse_limitations(ground_truth)
    gen_lims = parse_limitations(generated)

    if not gt_lims or not gen_lims:
        return 0.0

    matched_gt = 0
    for gt in gt_lims:
        max_sim = 0.0
        for gen in gen_lims:
            try:
                vect = TfidfVectorizer().fit([gt, gen])
                tfidf = vect.transform([gt, gen])
                sim = cosine_similarity(tfidf[0], tfidf[1])[0, 0]
                max_sim = max(max_sim, sim)
            except Exception:
                pass
        if max_sim > 0.3:  # Threshold for "matched"
            matched_gt += 1

    return matched_gt / len(gt_lims)

def compute_precision(generated: str, ground_truth: str) -> float:
    """
    Precision: What fraction of generated limitations are valid?
    A generated limitation is "valid" if it matches any GT limitation.
    """
    gt_lims = parse_limitations(ground_truth)
    gen_lims = parse_limitations(generated)

    if not gen_lims:
        return 0.0
    if not gt_lims:
        return 0.0

    valid_gen = 0
    for gen in gen_lims:
        max_sim = 0.0
        for gt in gt_lims:
            try:
                vect = TfidfVectorizer().fit([gt, gen])
                tfidf = vect.transform([gt, gen])
                sim = cosine_similarity(tfidf[0], tfidf[1])[0, 0]
                max_sim = max(max_sim, sim)
            except Exception:
                pass
        if max_sim > 0.3:
            valid_gen += 1

    return valid_gen / len(gen_lims)

def compute_groundedness(generated: str, paper_text: str) -> float:
    """
    Groundedness: Are evidence pointers in the generated text real?
    Checks for references to specific sections, tables, figures, etc.
    """
    refs = len(re.findall(
        r"(section|table|figure|equation|experiment|appendix|algorithm)\s*\w*",
        generated, re.IGNORECASE
    ))
    quotes = len(re.findall(r'"[^"]{5,}"', generated))
    return min(1.0, refs * 0.12 + quotes * 0.08)

def compute_redundancy_penalty(generated: str) -> float:
    """
    Redundancy penalty: Penalize repetitive limitations.
    Returns a value between 0 (no redundancy) and 1 (very redundant).
    """
    lims = parse_limitations(generated)
    if len(lims) <= 1:
        return 0.0

    redundant_pairs = 0
    total_pairs = 0
    for i in range(len(lims)):
        for j in range(i + 1, len(lims)):
            total_pairs += 1
            try:
                vect = TfidfVectorizer().fit([lims[i], lims[j]])
                tfidf = vect.transform([lims[i], lims[j]])
                sim = cosine_similarity(tfidf[0], tfidf[1])[0, 0]
                if sim > 0.7:  # High similarity = redundant
                    redundant_pairs += 1
            except Exception:
                pass

    return redundant_pairs / max(total_pairs, 1)

def compute_specificity(generated: str) -> float:
    """
    Specificity: Does the text contain specific, evidence-grounded claims?
    """
    # Specific indicators
    specific_patterns = [
        r"\d+\.?\d*\s*%",          # Percentages
        r"table\s*\d+",            # Table references
        r"figure\s*\d+",           # Figure references
        r"section\s*\d+",          # Section references
        r"equation\s*\d+",         # Equation references
        r"\d+\.\d+",              # Decimal numbers
    ]
    hits = sum(
        1 for p in specific_patterns
        if re.search(p, generated, re.IGNORECASE)
    )
    return min(1.0, hits / 3)

def compute_criticality(generated: str) -> float:
    """
    Criticality: Assertive, critical language indicating real limitations.
    """
    words = [
        "fail", "lack", "miss", "overlook", "insufficient", "narrow", "weak",
        "mislead", "overstat", "ignor", "cherry.pick", "absent", "incomplete",
        "limit", "deficien", "inadequa", "flaw", "concern", "problematic",
    ]
    hits = sum(1 for w in words if re.search(w, generated, re.IGNORECASE))
    return min(1.0, hits / 5)

def compute_meteor(generated: str, ground_truth: str) -> float:
    """METEOR score between generated and ground truth."""
    if not HAS_NLTK or not generated or not ground_truth:
        return 0.0
    try:
        hyp = nltk.word_tokenize(generated.lower())
        ref = nltk.word_tokenize(ground_truth[:3000].lower())
        score = nltk_meteor([ref], hyp)
        return min(1.0, score / 0.4)  # Normalize
    except Exception:
        return 0.0

def rule_based_reward(
    generated: str,
    ground_truth: str,
    paper_text: str,
    config: RewardConfig,
) -> Dict[str, float]:
    """
    Compute rule-based reward combining all NLP metrics.
    Returns individual scores and weighted total.
    """
    scores = {
        "coverage": compute_coverage(generated, ground_truth),
        "precision": compute_precision(generated, ground_truth),
        "groundedness": compute_groundedness(generated, paper_text),
        "redundancy": compute_redundancy_penalty(generated),
        "specificity": compute_specificity(generated),
        "criticality": compute_criticality(generated),
    }

    # Also compute METEOR if available
    if HAS_NLTK:
        scores["meteor"] = compute_meteor(generated, ground_truth)

    # Weighted combination
    weights = {
        "coverage": config.coverage_weight,
        "precision": config.precision_weight,
        "groundedness": config.groundedness_weight,
        "redundancy": -config.redundancy_penalty_weight,  # Negative = penalty
        "specificity": config.specificity_weight,
        "criticality": config.criticality_weight,
    }

    total_weight = sum(abs(w) for w in weights.values())
    weighted_score = sum(scores[k] * weights.get(k, 0) for k in scores if k in weights)
    weighted_score /= total_weight

    scores["rule_based_total"] = max(0.0, weighted_score)  # Clamp to non-negative
    return scores

# ================================================================
# 2. ZERO-SHOT LLM REWARD
# ================================================================

def zero_shot_reward(
    generated: str,
    ground_truth: str,
    model,
    tokenizer,
    device: str = "cuda",
) -> float:
    """
    Use the model itself as a zero-shot judge.
    Ask it to rate the quality of generated limitations vs ground truth.
    Returns a score between 0 and 1.
    """
    judge_prompt = f"""Rate the quality of the generated limitations compared to the ground truth.

Ground Truth Limitations:
{ground_truth[:1500]}

Generated Limitations:
{generated[:1500]}

Rate on a scale of 1-10 where:
1-3: Poor (misses most key limitations, many false positives)
4-6: Moderate (captures some limitations but misses important ones)
7-9: Good (captures most limitations with good specificity)
10: Excellent (comprehensive, specific, well-organized)

Respond with ONLY a single number (1-10)."""

    messages = [
        {"role": "system", "content": "You are an expert research paper reviewer evaluating limitation analysis quality."},
        {"role": "user", "content": judge_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=3500)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Parse score
    try:
        # Extract first number from response
        match = re.search(r"(\d+)", response)
        if match:
            score = int(match.group(1))
            return min(1.0, max(0.0, score / 10.0))
    except Exception:
        pass

    return 0.5  # Default if parsing fails

# ================================================================
# 3. TRAINED REWARD MODEL
# ================================================================

class RewardModelHead(nn.Module):
    """
    Reward model head: takes the last hidden state and produces a scalar reward.
    Attached on top of the SFT model.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Use the last token's hidden state
        return self.head(hidden_states[:, -1, :]).squeeze(-1)

class TrainedRewardModel:
    """
    Reward model fine-tuned from SFT checkpoint.
    Trained on preference pairs (chosen/rejected) derived from
    comparing generated limitations with ground truth.
    """

    def __init__(self, model, tokenizer, device, hidden_size=2048):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.reward_head = RewardModelHead(hidden_size).to(device)
        self.is_trained = False

    def compute_reward(self, text: str, paper_text: str) -> float:
        """Compute scalar reward for a generated limitation text."""
        if not self.is_trained:
            return 0.5  # Untrained default

        prompt = f"Rate the quality of these research paper limitations:\n\nPaper excerpt: {paper_text[:1000]}\n\nLimitations: {text[:1000]}"
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=2048,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            reward = self.reward_head(hidden)

        return torch.sigmoid(reward).item()

    def train_on_preferences(
        self,
        preference_pairs: List[Dict],
        epochs: int = 2,
        lr: float = 2e-5,
        batch_size: int = 4,
    ):
        """
        Train reward model on preference pairs.
        Each pair: {"chosen": str, "rejected": str, "paper": str}
        """
        log.info(f"Training reward model on {len(preference_pairs)} preference pairs")

        optimizer = torch.optim.AdamW(self.reward_head.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            n_batches = 0

            # Shuffle pairs
            import random
            random.shuffle(preference_pairs)

            for i in range(0, len(preference_pairs), batch_size):
                batch = preference_pairs[i:i + batch_size]

                chosen_rewards = []
                rejected_rewards = []

                for pair in batch:
                    # Compute rewards for chosen and rejected
                    paper = pair["paper"][:1000]

                    for text, target_list in [
                        (pair["chosen"], chosen_rewards),
                        (pair["rejected"], rejected_rewards),
                    ]:
                        prompt = f"Rate limitations:\nPaper: {paper}\nLimitations: {text[:1000]}"
                        messages = [{"role": "user", "content": prompt}]
                        input_text = self.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=False,
                        )
                        inputs = self.tokenizer(
                            input_text, return_tensors="pt",
                            truncation=True, max_length=2048,
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = self.model(**inputs, output_hidden_states=True)
                            hidden = outputs.hidden_states[-1]

                        reward = self.reward_head(hidden)
                        target_list.append(reward)

                if not chosen_rewards:
                    continue

                # Bradley-Terry loss: log sigmoid(r_chosen - r_rejected)
                chosen_r = torch.stack(chosen_rewards)
                rejected_r = torch.stack(rejected_rewards)
                loss = -torch.log(torch.sigmoid(chosen_r - rejected_r) + 1e-8).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (chosen_r > rejected_r).sum().item()
                n_batches += 1

            acc = correct / max(len(preference_pairs), 1) * 100
            avg_loss = total_loss / max(n_batches, 1)
            log.info(f"  Reward model epoch {epoch+1}/{epochs}: "
                     f"loss={avg_loss:.4f}, accuracy={acc:.1f}%")

        self.is_trained = True
        log.info("Reward model training complete")

    def save(self, path: str):
        torch.save(self.reward_head.state_dict(), path)
        log.info(f"Reward head saved to {path}")

    def load(self, path: str):
        self.reward_head.load_state_dict(torch.load(path, map_location=self.device))
        self.is_trained = True
        log.info(f"Reward head loaded from {path}")

# ================================================================
# 4. COMBINED REWARD SCORING
# ================================================================

def score_trajectory(
    generated: str,
    ground_truth: str,
    paper_text: str,
    config: RewardConfig,
    model=None,
    tokenizer=None,
    trained_reward_model: Optional[TrainedRewardModel] = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute combined reward for a generated limitation set.
    Combines rule-based, zero-shot, and trained reward signals.
    """
    scores = rule_based_reward(generated, ground_truth, paper_text, config)

    # Zero-shot LLM reward
    if config.use_zero_shot_reward and model is not None and tokenizer is not None:
        zs_score = zero_shot_reward(generated, ground_truth, model, tokenizer, device)
        scores["zero_shot"] = zs_score

    # Trained reward model
    if trained_reward_model is not None and trained_reward_model.is_trained:
        tr_score = trained_reward_model.compute_reward(generated, paper_text)
        scores["trained_reward"] = tr_score

    # Combined score
    total = scores["rule_based_total"]
    n_components = 1

    if "zero_shot" in scores:
        total += scores["zero_shot"] * config.zero_shot_weight
        n_components += config.zero_shot_weight

    if "trained_reward" in scores:
        total += scores["trained_reward"]
        n_components += 1

    scores["combined_reward"] = total / n_components
    return scores

def score_all_rollouts(
    all_trajectories: List[List],  # [paper_idx][rollout_idx]
    config: PipelineConfig,
    model=None,
    tokenizer=None,
    trained_reward_model=None,
) -> List[List[Dict]]:
    """
    Score all rollouts for all papers.
    Returns scores aligned with trajectories.
    """
    all_scores = []

    for paper_trajs in all_trajectories:
        paper_scores = []
        for traj in paper_trajs:
            if isinstance(traj, dict):
                generated = traj.get("final_limitations", "")
                gt = traj.get("ground_truth", "")
                paper = traj.get("paper_text", "")
            else:
                generated = traj.final_limitations
                gt = traj.ground_truth
                paper = traj.paper_text

            scores = score_trajectory(
                generated, gt, paper,
                config.reward,
                model=model,
                tokenizer=tokenizer,
                trained_reward_model=trained_reward_model,
            )
            paper_scores.append(scores)

        all_scores.append(paper_scores)

    return all_scores

# ================================================================
# 5. PREFERENCE PAIR GENERATION (for training reward model)
# ================================================================

def create_preference_pairs(
    all_trajectories: List[List],
    all_scores: List[List[Dict]],
    config: RewardConfig,
) -> List[Dict]:
    """
    Create preference pairs from scored rollouts.
    Within each paper's K rollouts, pair the best vs worst.
    """
    pairs = []

    for paper_idx, (trajs, scores) in enumerate(zip(all_trajectories, all_scores)):
        if len(trajs) < 2:
            continue

        # Sort by combined reward
        indexed = [(i, s["rule_based_total"]) for i, s in enumerate(scores)]
        indexed.sort(key=lambda x: x[1], reverse=True)

        best_idx = indexed[0][0]
        worst_idx = indexed[-1][0]
        best_score = indexed[0][1]
        worst_score = indexed[-1][1]

        # Only create pair if there's meaningful difference
        if best_score - worst_score < 0.1:
            continue

        best_traj = trajs[best_idx]
        worst_traj = trajs[worst_idx]

        if isinstance(best_traj, dict):
            chosen = best_traj.get("final_limitations", "")
            rejected = worst_traj.get("final_limitations", "")
            paper = best_traj.get("paper_text", "")[:2000]
        else:
            chosen = best_traj.final_limitations
            rejected = worst_traj.final_limitations
            paper = best_traj.paper_text[:2000]

        if chosen and rejected:
            pairs.append({
                "chosen": chosen,
                "rejected": rejected,
                "paper": paper,
                "chosen_score": best_score,
                "rejected_score": worst_score,
            })

    log.info(f"Created {len(pairs)} preference pairs")
    return pairs  
