import re
from spam_demo import calculate_hybrid_score, load_artifacts

TEST_CASES = [
    # ==========================================
    # RED TEAM (Expected: SPAM) - 高难度诈骗
    # ==========================================
    # 1. 股票内幕 (Stock Insider) - 之前漏判的重灾区
    ("My uncle has inside info on a stock that will explode 500% tomorrow. Buy now!", "SPAM"),
    ("Guaranteed return on investment. This tech stock is the next big thing.", "SPAM"),
    ("Insider tip: TSLA is going to skyrocket. Act fast before market closes.", "SPAM"),

    # 2. 洗钱/资金转移 (Money Mule) - 之前漏判
    ("I need help moving some funds to an overseas account. You keep 10% commission.", "SPAM"),
    ("Transaction assistance needed. Easy job, keep a cut of the money.", "SPAM"),

    # 3. 裸聊/勒索 (Sextortion) - 之前权重不足
    ("I hacked your cloud storage. I have your private videos. Pay me or I leak them.", "SPAM"),
    ("I recorded you accessing adult sites. Send bitcoin to silence me.", "SPAM"),
    ("Pay $500 to keep your private photos secret. Otherwise sending to all contacts.", "SPAM"),

    # 4. 加密货币 (Crypto) - 各种变体
    ("USDT Airdrop event is live! Connect your wallet to claim free tokens.", "SPAM"),
    ("Click bit.ly/claim to verify your wallet and stop suspension.", "SPAM"),

    # 5. 变体/混淆 (Obfuscation)
    ("W0rk fr0m h0me, e@rn $500 da!ly. No experience needed.", "SPAM"),
    ("Urgent: Y0ur acc0unt is l0cked. Verify identity n0w.", "SPAM"),
    ("FedEx: Delivery pending. Pay small fee at track-package.com/pay", "SPAM"),
    ("Congratulations! You w0n a $1000 Walmart gift card.", "SPAM"),
    ("Hiring: Simple texting job. No exp req. Daily payout.", "SPAM"),

    # ==========================================
    # BLUE TEAM (Expected: HAM) - 极易误判的正常短信
    # ==========================================
    # 1. 抱怨/负面情绪 (Complaint) - 容易被误认为涉及金钱交易
    ("I am so sad, I lost all my money investing in bitcoin last year.", "HAM"),
    ("My stock portfolio crashed today. I hate this market.", "HAM"),
    ("I lost my wallet on the bus. Can you pick me up?", "HAM"),

    # 2. 正常工作/请假 (Work Context) - 容易触发兼职诈骗规则
    ("I'm not feeling well today, so I will work from home and check emails.", "HAM"),
    ("Can you send me the Zoom link for the meeting?", "HAM"),
    ("Here is the project proposal: docs.google.com/presentation/d/xyz", "HAM"),

    # 3. 还钱/借贷 (Repayment) - 容易触发 Money 规则
    ("Thanks for dinner! I just transferred the $50 I owed you.", "HAM"),
    ("Hey, can I borrow $10 for lunch? I forgot my cash.", "HAM"),
    ("I will return the money I borrowed next week.", "HAM"),

    # 4. 新闻/讨论 (News) - 提及敏感词但无恶意
    ("Did you see the news that Elon Musk is buying Twitter?", "HAM"),
    ("We need to discuss the budget for the marketing campaign.", "HAM"),

    # 5. 其他干扰项
    ("Stop sending me spam messages or I will block you!", "HAM"),
    ("Your verification code is 123456. Do not share it.", "HAM"),
    ("Happy birthday! I hope you have a great year ahead.", "HAM"),
    ("Where are you? We are waiting at the restaurant.", "HAM")
]


def classify(score):
    return "SPAM" if score > 0.5 else "HAM"


def categorize(text, expected):
    t = text.lower()
    if expected == "HAM":
        if any(x in t for x in ["docs.google", "drive.google", "zoom.us", "teams.microsoft", "github.com"]):
            return "ham_office_link"
        if any(x in t for x in ["owed", "borrowed", "repay", "return money", "owe"]):
            return "ham_repayment"
        if any(x in t for x in ["lost money", "regret", "sad", "bad"]):
            return "ham_complaint"
        if "work from home" in t:
            return "ham_wfh"
        return "ham_other"

    if any(x in t for x in ["private video", "private photos", "leak", "hacked", "cloud storage"]):
        return "spam_sextortion"
    if any(x in t for x in ["inside info", "insider tip", "stock explode", "guaranteed return"]):
        return "spam_stock"
    if any(x in t for x in ["transaction assistance", "move funds", "commission"]):
        return "spam_money_mule"
    if any(x in t for x in ["btc", "bitcoin", "usdt", "crypto"]):
        return "spam_crypto"
    return "spam_other"


def run_audit():
    load_artifacts()
    total = len(TEST_CASES)
    correct = 0
    group_stats = {}

    for text, expected in TEST_CASES:
        score, hits = calculate_hybrid_score(text, 0.0)
        got = classify(score)
        group = categorize(text, expected)
        if group not in group_stats:
            group_stats[group] = {"total": 0, "correct": 0}
        group_stats[group]["total"] += 1
        if got == expected:
            correct += 1
            group_stats[group]["correct"] += 1
        else:
            print(f"[FAILURE] Input: \"{text}\" | Expected: {expected} | Got: {got} | Triggers: {hits}")

    acc = (correct / total) * 100
    print(f"Accuracy: {acc:.2f}% ({correct}/{total})")
    for group in sorted(group_stats.keys()):
        g_total = group_stats[group]["total"]
        g_correct = group_stats[group]["correct"]
        g_acc = (g_correct / g_total) * 100 if g_total else 0.0
        print(f"{group}: {g_acc:.2f}% ({g_correct}/{g_total})")


if __name__ == "__main__":
    run_audit()
