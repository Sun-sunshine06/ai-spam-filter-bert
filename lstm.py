import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("正在初始化系统...")

# === 1. 参数升级 ===
# 扩大词表，确保能认识 "asshole", "shit" 等非核心词汇
vocab_size = 20000
max_len = 150

print("正在加载/下载 IMDB 数据集...")
try:
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
except:
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)

# 数据预处理
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

print(f"数据准备完毕！词表大小: {vocab_size}")

# === 2. 模型升级 (Deep Bi-LSTM) ===
model = keras.Sequential([
    # 维度扩大到 64
    layers.Embedding(input_dim=vocab_size, output_dim=64),

    # 双层 LSTM 结构：模拟更深层的思考
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),  # 第一层传给第二层
    layers.Dropout(0.3),
    layers.Bidirectional(layers.LSTM(32)),  # 第二层输出结果

    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# === 3. 训练 (可能比之前慢一点点，但效果更好) ===
print("\n开始训练模型...")
history = model.fit(x_train, y_train,
                    epochs=3,
                    batch_size=128,
                    validation_split=0.2)

# === 4. 评估 ===
print("\n正在评估测试集...")
results = model.evaluate(x_test, y_test)
print(f"最终准确率: {results[1] * 100:.2f}%")

# 获取词表索引
word_index = keras.datasets.imdb.get_word_index()


# === 5. 交互演示 ===
def predict_interactive():
    print("\n" + "=" * 50)
    print("🎬 AI 影评情感分析系统")
    print("👉 输入 'exit' 退出")
    print("=" * 50)

    while True:
        text = input("\n👉 请输入影评: ")

        if text.lower() in ['exit', 'quit']:
            break

        if not text.strip():
            continue

        # 预处理
        text_clean = text.lower().replace(",", "").replace(".", "").replace("!", "").replace("?", "")
        words = text_clean.split()

        review = [1]
        for word in words:
            if word in word_index and (word_index[word] + 3) < vocab_size:
                review.append(word_index[word] + 3)
            else:
                review.append(2)  # 未知词

        # 变长输入
        review = keras.preprocessing.sequence.pad_sequences([review])

        prediction = model.predict(review, verbose=0)[0][0]

        # 可视化
        bar_len = 20
        filled_len = int(bar_len * prediction)
        filled_len = max(0, min(bar_len, filled_len))
        bar = '█' * filled_len + '░' * (bar_len - filled_len)
        score_percent = prediction * 100

        if 0.45 <= prediction <= 0.55:
            sentiment = "🤔 语气不确定 (Neutral)"
            color_code = "\033[93m"
        elif prediction > 0.55:
            sentiment = "😊 正面好评 (Positive)"
            color_code = "\033[92m"
        else:
            sentiment = "😡 负面差评 (Negative)"
            color_code = "\033[91m"

        reset_code = "\033[0m"

        print(f"   --------------------------------------------------")
        print(f"   {color_code}判定结果: {sentiment}{reset_code}")
        print(f"   情感置信度: [{bar}] {score_percent:.2f}%")
        print(f"   --------------------------------------------------")


predict_interactive()