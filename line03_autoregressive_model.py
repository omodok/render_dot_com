#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os  # osモジュールをインポート
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from omodok_transformer import * 


# In[2]:


# 自己回帰トレーニング
def ar_training(decoder, linear_layer, embedding_layer, dataloader, 
                vocab, epochs=30, lr=0.001, save_path="model.pth"):
    # モデルが既に存在する場合、学習をスキップしてモデルを読み込む
    if os.path.exists(save_path):
        print(f"モデル '{save_path}' が見つかりました。学習をスキップしてモデルを読み込みます。")
        checkpoint = torch.load(save_path)
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        linear_layer.load_state_dict(checkpoint['linear_layer_state_dict'])
        embedding_layer.load_state_dict(checkpoint['embedding_layer_state_dict'])
        return decoder, linear_layer, embedding_layer
    
    # 損失関数とオプティマイザ
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    optimizer = optim.Adam(list(decoder.parameters()) + list(linear_layer.parameters()), lr=lr)

    # 学習ループ
    for epoch in range(epochs):
        total_loss = 0
        for input_tensor, target_tensor in dataloader:
            optimizer.zero_grad()

            # 出題側の埋め込み
            embedded_input = embedding_layer(input_tensor)

            # マスク配列の作成
            seq_len = embedded_input.size(1)  # デコーダーの入力シーケンス長を取得
            tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)  # 下三角行列）

            # デコーダーの実行
            decoder_output = decoder(embedded_input, tgt_mask)

            # 全出力トークンを線形層に渡す
            logits = linear_layer(decoder_output)

            # logits の形状を (batch_size * seq_len, vocab_size) に整形
            logits_reshaped = logits.reshape(-1, vocab_size)

            # target_tensor の形状を (batch_size * seq_len) に整形
            tgt_reshaped = target_tensor.reshape(-1)

            # 損失の計算
            loss = criterion(logits_reshaped, tgt_reshaped)

            # 誤差逆伝播とパラメータ更新
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")
    
    # モデルを保存
    torch.save({
        'decoder_state_dict': decoder.state_dict(),
        'linear_layer_state_dict': linear_layer.state_dict(),
        'embedding_layer_state_dict': embedding_layer.state_dict(),
        'vocab': vocab
    }, save_path)
    print(f"モデルを '{save_path}' に保存しました。")

    return decoder, linear_layer, embedding_layer

# データセットの素材作成関数
def prepare_inputs_labels(train_data, vocab, max_len=50):
    inputs = []
    labels = []
    
    for text in train_data:
        # テキストのトークン化とインデックス化
        tokens = [vocab['<start>']]
        tokens += [vocab.get(token, vocab['<unk>']) for token in tokenize(text)]
        tokens += [vocab['<end>']]
        
        # パディングを行い、固定長のシーケンスにする
        if len(tokens) < max_len:
            tokens += [vocab['<pad>']] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        
        # 出題側と正解側に分割
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        
        inputs.append(input_tokens)
        labels.append(target_tokens)
    
    # テンソル化
    inputs_tensor = torch.tensor(inputs)
    labels_tensor = torch.tensor(labels)
    
    return inputs_tensor, labels_tensor


# In[3]:


# 推論
def generate_text(decoder, linear_layer, embedding_layer, vocab, prompt, max_output_length=40):

    decoder.eval()  # モデルを評価モードに設定

    tokens = [vocab['<start>']] 
    tokens += [vocab.get(token, vocab['<unk>']) for token in tokenize(prompt)]
    decoder_input = torch.tensor(tokens).unsqueeze(0)

    output_sequence = []   #生成されたトークンを格納する配列

    for _ in range(max_output_length):
        # 埋め込み層を作成
        embedded_input = embedding_layer(decoder_input)

        # マスク配列の作成
        seq_len = embedded_input.size(1)
        tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)

        # デコーダーを実行
        decoder_output = decoder(embedded_input, tgt_mask)

        # 最後のトークンのみを線形層に渡す
        logits = linear_layer(decoder_output[:, -1, :])  # (batch_size, vocab_size)

        # ソフトマックスを適用してトークンの確率を計算
        output_probabilities = torch.softmax(logits, dim=-1)

        # 最も確率の高いトークンを予測（argmaxで選択）
        predicted_token = output_probabilities.argmax(dim=-1)  # (batch_size, 1)

        # 予測したトークンを出力シーケンスに追加
        output_sequence.append(predicted_token.item())  # item()でテンソルからスカラーを取得

        # 次のステップに向けてデコーダー入力を更新（自己回帰的に）
        decoder_input = torch.cat([decoder_input, predicted_token.unsqueeze(1)], dim=1)

        # <end> トークンが予測されたら終了（大規模言語モデルが使用）
        if predicted_token.item() == vocab['<end>']:
            break
            
    return output_sequence


# In[4]:


##### メイン処理
# 学習データの準備
train_data = [
    "太郎は花子が好きだと言った。",
    "今日は晴れです。",
    "明日は雨が降るかもしれません。",
    "私はプログラミングが好きです。",
    "彼は本を読むのが趣味ですか。",
    "猫は魚が好きです。",
    "犬は散歩が大好きです。",
    "東京は日本の首都です。",
    "京都は歴史的な街です。",
    "寿司は日本の伝統料理です。",
    "ラーメンは人気のある料理です。"
]

def tokenize(text):
    """MeCabを使わない単語トークンに分割"""
    #return mecab.parse(text).strip().split()
    return list(text)

# 語彙の初期化
vocab = {"<start>": 0, "<end>": 1, "<pad>": 2, "<unk>": 3}
for text in train_data:
    for token in tokenize(text):
        if token not in vocab:
            vocab[token] = len(vocab)
            
vocab_size = len(vocab)    # 語彙の総数
start_token_index = vocab['<start>']    # 特殊トークンインデックスを取得
end_token_index = vocab['<end>']


# In[5]:


# 埋め込み層の定義
embedding_dim = 256  # 埋め込み次元

# ターゲット用の埋め込みレイヤー
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# モデルのパラメータ
embed_size = embedding_dim
heads = 8
ff_dim = embed_size * 2
num_layers = 6
max_len = 50
dropout = 0.1

# GPT風デコーダーのインスタンス化
decoder = GPTStyleDecoder(embed_size, heads, ff_dim, num_layers, max_len, dropout)

# デコーダが使用する線形変換
linear_layer = nn.Linear(embed_size, vocab_size)  # 出力次元を vocab_size に合わせる

# データセットの準備
inputs, labels = prepare_inputs_labels(train_data, vocab)

#データセット化
dataset = TensorDataset(inputs, labels) 
dataloader = DataLoader(dataset)

# 学習またはモデルの読み込み
decoder, linear_layer, embedding_layer = ar_training(
    decoder, linear_layer, embedding_layer, dataloader, vocab, epochs=30, save_path="none_mecab_model.pth")


# In[6]:


# vocab 辞書を反転させてインデックスからトークンに変換できるようにする
reverse_vocab = {idx: token for token, idx in vocab.items()}  # tgt_vocabに変える

# プロンプトを入力して推論
prompt = input("プロンプトを入力してください: ")
output_sequence = generate_text(decoder, linear_layer, embedding_layer, vocab, prompt)

# 出力トークンをトークンに変換
output_tokens_as_text = [reverse_vocab[token_idx] for token_idx in output_sequence]

# 結果を表示
print("生成された出力トークン:","".join(output_tokens_as_text).replace("<end>", ""))


# In[ ]:




