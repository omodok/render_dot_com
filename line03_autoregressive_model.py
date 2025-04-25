#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os  # osモジュールをインポート
import MeCab
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        """
        コンストラクタ定義文
        第1引き数：embed_size: 埋め込みベクトルの次元数
        第2引き数：heads: アテンションヘッドの数
        """
        super(SelfAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # 線形変換層の定義
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask=None):
        """
        順伝播関数定義文
        第1引き数：values: 値 (batch_size, value_len, embed_size)
        第2引き数：keys: キー (batch_size, key_len, embed_size)
        第3引き数：query: クエリ (batch_size, query_len, embed_size)
        第4引き数：mask: マスク (batch_size, query_len, key_len)
        戻り値:return: アテンションの出力 (batch_size, query_len, embed_size)
        """
        # バッチサイズとシーケンス長を取得
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # テンソルの形状を変更 (batch_size, seq_len, heads, head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # 線形変換の適用
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # アテンションスコアの計算
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # torch.Size([1, 8, 13, 13])

        # マスクを適用（オプション）
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # ソフトマックスを適用
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # torch.Size([1, 8, 13, 13])

        #アテンションの出力（加重平均）を計算する
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])  # torch.Size([1, 13, 8, 32])
        
        #ヘッドを結合し、テンソル形状を戻す
        out = out.reshape(N, query_len, self.heads * self.head_dim)  # torch.Size([1, 13, 256])

        # 最終的な線形変換を適用
        out = self.fc_out(out)  # torch.Size([1, 13, 256])
        
        return out


# In[3]:


# AddNormクラスの定義
class AddNorm(nn.Module):
    def __init__(self, embed_size):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(embed_size)  # レイヤー正規化

    def forward(self, x, sublayer_output):
        # 残差接続 (Add)
        out = x + sublayer_output
        # レイヤー正規化 (Norm)
        out = self.norm(out)
        return out


# In[4]:


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, ff_dim, dropout):
        """
        コンストラクタ定義文
        第1引き数：embed_size: 埋め込みベクトルの次元数
        第2引き数：ff_dim: フィードフォワードネットワークの隠れ層の次元数
        第3引き数：dropout: ドロップアウト率        
        """
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        順伝播関数定義文
        第1引き数：x: 入力テンソル (batch_size, seq_len, embed_size)
        戻り値: フィードフォワードネットワークの出力 (batch_size, seq_len, embed_size)
        """
        out = self.ff(x)        # フィードフォワードネットワークを適用
        return out


# In[5]:


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        """
        コンストラクタ定義文
        第1引き数：embed_size: 埋め込みベクトルの次元数
        第2引き数：max_len: 最大シーケンス長
        """
        super(PositionalEncoding, self).__init__()

        # 位置エンコーディング用の空のテンソルを3つ作成
        pe = torch.zeros(max_len, embed_size)      #位置埋め込みベクトルの作成
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 分数式の分子作成
        log_term = -torch.log(torch.tensor(10000.0)) / embed_size   # 定数部分の計算
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * log_term)   # 分数式の分母作成

        # 正弦波と余弦波を交互に設定
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数次元: 正弦波
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数次元: 余弦波

        # バッチ次元を追加
        pe = pe.unsqueeze(0)
        
        # 学習対象ではないため、バッファとして登録
        self.register_buffer('pe', pe) 
                
    def forward(self, x):
        """
        順伝播関数定義文
        第1引き数：x: 入力テンソル (batch_size, seq_len, embed_size)
        戻り値: 位置エンコーディングを追加したテンソル (batch_size, seq_len, embed_size)
        """
        if x.size(1)<=self.pe.size(1):
            x = x + self.pe[:, :x.size(1)]  # 位置エンコーディングを追加
        return x


# In[6]:


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_dim, dropout=0.1):
        """
        コンストラクタ定義文
        第1引き数：embed_size: 埋め込みベクトルの次元数
        第2引き数：heads: アテンションヘッドの数
        第3引き数：ff_dim: フィードフォワードネットワークの隠れ層の次元数
        第4引き数：dropout: ドロップアウト率
        """
        super(EncoderLayer, self).__init__()
        
        # 「Self Attention」のインスタンス化
        self.self_attention = SelfAttention(embed_size, heads)
        
        # Add & Norm 1のインスタンス化
        self.add_norm1 = AddNorm(embed_size)
        
        # FeedForwardのインスタンス化
        self.feed_forward = FeedForward(embed_size, ff_dim, dropout)
        
        # Add & Norm 2のインスタンス化
        self.add_norm2 = AddNorm(embed_size)
        
        # ドロップアウトのインスタンス化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        順伝播関数定義文
        第1引き数：x: 入力テンソル (batch_size, seq_len, embed_size)
        第2引き数：mask: マスク (batch_size, seq_len, seq_len)
        戻り値: エンコーダーレイヤーの出力 (batch_size, seq_len, embed_size)
        """
        # 「Self Attention」の実行
        attention_output = self.self_attention(x, x, x, mask)
        attention_output = self.dropout(attention_output)
        
        # Add & Norm 1の実行
        out = self.add_norm1(x, attention_output)
        
        # FeedForwardの実行
        ff_output = self.feed_forward(out)
        ff_output = self.dropout(ff_output)
        
        # Add & Norm 2の実行
        out = self.add_norm2(out, ff_output)
        
        return out


# In[7]:


class Encoder(nn.Module):
    def __init__(self, embed_size, heads, ff_dim, num_layers, max_len=5000, dropout=0.1):
        """
        コンストラクタ定義文
        第1引き数：embed_size: 埋め込みベクトルの次元数
        第2引き数：heads: アテンションヘッドの数
        第3引き数：ff_dim: フィードフォワードネットワークの隠れ層の次元数
        第4引き数：num_layers: エンコーダーレイヤーの数 (N)
        第5引き数：max_len: 最大シーケンス長
        第6引き数：dropout: ドロップアウト率
        """
        super(Encoder, self).__init__()
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(embed_size, max_len)
        
        # エンコーダーレイヤーを num_layers 回繰り返す
        self.layers = nn.ModuleList([
            EncoderLayer(embed_size, heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        順伝播関数定義文
        第1引き数：x: 入力テンソル (batch_size, seq_len, embed_size)
        第2引き数：mask: マスク (batch_size, seq_len, seq_len)
        戻り値: エンコーダーの出力 (batch_size, seq_len, embed_size)
        """
        # Positional Encoding の実行
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # エンコーダーレイヤーを num_layers 回実行
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


# In[8]:


class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MaskedSelfAttention, self).__init__()
        self.self_attention = SelfAttention(embed_size, heads)

    def forward(self, x, mask):
        """
        順伝播関数定義文
        第1引き数：x: 入力テンソル (batch_size, seq_len, embed_size)
        第2引き数：mask: マスク (batch_size, seq_len, seq_len)
        戻り値: マスク付きセルフアテンションの出力 (batch_size, seq_len, embed_size)
        """
        return self.self_attention(x, x, x, mask)


# In[9]:


class EncoderDecoderAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(EncoderDecoderAttention, self).__init__()
        self.attention = SelfAttention(embed_size, heads)

    def forward(self, x, encoder_output, mask=None):
        """
        順伝播関数定義文
        第1引き数：x: デコーダーの入力テンソル (batch_size, seq_len, embed_size)
        第2引き数：encoder_output: エンコーダーの出力テンソル (batch_size, seq_len, embed_size)
        第3引き数：mask: マスク (batch_size, seq_len, seq_len)
        戻り値: エンコーダー-デコーダーアテンションの出力 (batch_size, seq_len, embed_size)
        """
        return self.attention(encoder_output, encoder_output, x, mask)


# In[10]:


class DecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_dim, dropout=0.1):
        """
        コンストラクタ定義文
        第1引き数：embed_size: 埋め込みベクトルの次元数
        第2引き数：heads: アテンションヘッドの数
        第3引き数：ff_dim: フィードフォワードネットワークの隠れ層の次元数
        第4引き数：dropout: ドロップアウト率
        """
        super(DecoderLayer, self).__init__()

        # マスク付きセルフアテンションのインスタンス化
        self.masked_self_attention = MaskedSelfAttention(embed_size, heads)
        # add_norm1のインスタンス化
        self.add_norm1 = AddNorm(embed_size)
        # エンコーダー-デコーダーアテンションのインスタンス化
        self.encoder_decoder_attention = EncoderDecoderAttention(embed_size, heads)
        # add_norm2のインスタンス化
        self.add_norm2 = AddNorm(embed_size)
        # フィードフォワードネットワークのインスタンス化
        self.feed_forward = FeedForward(embed_size, ff_dim, dropout)
        # add_norm3のインスタンス化
        self.add_norm3 = AddNorm(embed_size)
        # ドロップアウトのインスタンス化
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        順伝播関数定義文
        第1引き数：x: デコーダーの入力テンソル (batch_size, seq_len, embed_size)
        第2引き数：encoder_output: エンコーダーの出力テンソル (batch_size, seq_len, embed_size)
        第3引き数：src_mask: エンコーダーのマスク (batch_size, seq_len, seq_len)
        第4引き数：tgt_mask: デコーダーのマスク (batch_size, seq_len, seq_len)
        戻り値: デコーダーレイヤーの出力 (batch_size, seq_len, embed_size)
        """
        # マスク付きセルフアテンションの実行
        masked_attention_output = self.masked_self_attention(x, tgt_mask)
        masked_attention_output = self.dropout(masked_attention_output)
        # add_norm1の実行
        out = self.add_norm1(x, masked_attention_output)
        # エンコーダー-デコーダーアテンションの実行
        encoder_decoder_output = self.encoder_decoder_attention(out, encoder_output, src_mask)
        encoder_decoder_output = self.dropout(encoder_decoder_output)
        # add_norm2の実行
        out = self.add_norm2(out, encoder_decoder_output)
        # フィードフォワードネットワークの実行
        ff_output = self.feed_forward(out)
        ff_output = self.dropout(ff_output)
        # add_norm3の実行
        out = self.add_norm3(out, ff_output)
        
        return out       


# In[11]:


class Decoder(nn.Module):
    def __init__(self, embed_size, heads, ff_dim, num_layers, max_len=5000, dropout=0.1):
        """
        コンストラクタ定義文
        第1引き数：embed_size: 埋め込みベクトルの次元数
        第2引き数：heads: アテンションヘッドの数
        第3引き数：ff_dim: フィードフォワードネットワークの隠れ層の次元数
        第4引き数：num_layers: デコーダーレイヤーの数 (N)
        第5引き数：max_len: 最大シーケンス長
        第6引き数：dropout: ドロップアウト率
        """
        super(Decoder, self).__init__()
        
        # 位置エンコーディングのインスタンス化
        self.positional_encoding = PositionalEncoding(embed_size, max_len)
        # デコーダーレイヤーを num_layers 回繰り返す
        self.layers = nn.ModuleList([
            DecoderLayer(embed_size, heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        # ドロップアウトのインスタンス化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        順伝播関数定義文
        第1引き数：x: デコーダーの入力テンソル (batch_size, seq_len, embed_size)
        第2引き数：encoder_output: エンコーダーの出力テンソル (batch_size, seq_len, embed_size)
        第3引き数：src_mask: エンコーダーのマスク (batch_size, seq_len, seq_len)
        第4引き数：tgt_mask: デコーダーのマスク (batch_size, seq_len, seq_len)
        戻り値: デコーダーの出力 (batch_size, seq_len, embed_size)
        """
        # 位置エンコーディングの実行
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # デコーダーレイヤーを num_layers 回実行
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x

class GPTStyleDecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_dim, dropout=0.1):
        """
        コンストラクタ定義文
        第1引き数：embed_size: 埋め込みベクトルの次元数
        第2引き数：heads: アテンションヘッドの数
        第3引き数：ff_dim: フィードフォワードネットワークの隠れ層の次元数
        第4引き数：dropout: ドロップアウト率
        """
        super(GPTStyleDecoderLayer, self).__init__()

        # マスク付きセルフアテンションのインスタンス化
        self.masked_self_attention = MaskedSelfAttention(embed_size, heads)
        # add_norm1のインスタンス化
        self.add_norm1 = AddNorm(embed_size)
        # フィードフォワードネットワークのインスタンス化
        self.feed_forward = FeedForward(embed_size, ff_dim, dropout)
        # add_norm2のインスタンス化
        self.add_norm2 = AddNorm(embed_size)
        # ドロップアウトのインスタンス化
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, tgt_mask):
        """
        順伝播関数定義文
        第1引き数：x: デコーダーの入力テンソル (batch_size, seq_len, embed_size)
        第2引き数：tgt_mask: デコーダーのマスク (batch_size, seq_len, seq_len)
        戻り値: デコーダーレイヤーの出力 (batch_size, seq_len, embed_size)
        """
        # マスク付きセルフアテンションの実行
        masked_attention_output = self.masked_self_attention(x, tgt_mask)
        masked_attention_output = self.dropout(masked_attention_output)
        # add_norm1の実行
        out = self.add_norm1(x, masked_attention_output)
        # フィードフォワードネットワークの実行
        ff_output = self.feed_forward(out)
        ff_output = self.dropout(ff_output)
        # add_norm2の実行
        out = self.add_norm2(out, ff_output)
        
        return out       


# In[11]:


class GPTStyleDecoder(nn.Module):
    def __init__(self, embed_size, heads, ff_dim, num_layers, max_len=5000, dropout=0.1):
        """
        コンストラクタ定義文
        第1引き数：embed_size: 埋め込みベクトルの次元数
        第2引き数：heads: アテンションヘッドの数
        第3引き数：ff_dim: フィードフォワードネットワークの隠れ層の次元数
        第4引き数：num_layers: デコーダーレイヤーの数 (N)
        第5引き数：max_len: 最大シーケンス長
        第6引き数：dropout: ドロップアウト率
        """
        super(GPTStyleDecoder, self).__init__()
        
        # 位置エンコーディングのインスタンス化
        self.positional_encoding = PositionalEncoding(embed_size, max_len)
        # デコーダーレイヤーを num_layers 回繰り返す
        self.layers = nn.ModuleList([
            GPTStyleDecoderLayer(embed_size, heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        # ドロップアウトのインスタンス化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tgt_mask):
        """
        順伝播関数定義文
        第1引き数：x: デコーダーの入力テンソル (batch_size, seq_len, embed_size)
        第2引き数：tgt_mask: デコーダーのマスク (batch_size, seq_len, seq_len)
        戻り値: デコーダーの出力 (batch_size, seq_len, embed_size)
        """
        # 位置エンコーディングの実行
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # デコーダーレイヤーを num_layers 回実行
        for layer in self.layers:
            x = layer(x, tgt_mask)
        
        return x

# In[2]:


# 自己回帰トレーニング
def ar_training(decoder, linear_layer, embedding_layer, dataloader, 
                vocab, epochs=30, lr=0.001, save_path="model.pth"):
    # モデルが既に存在する場合、学習をスキップしてモデルを読み込む
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        linear_layer.load_state_dict(checkpoint['linear_layer_state_dict'])
        embedding_layer.load_state_dict(checkpoint['embedding_layer_state_dict'])
        print(f"モデル '{save_path}' が見つかりました。学習をスキップしてモデルを読み込みます。")
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

#mecab = MeCab.Tagger("-Owakati")   #分かち書きオプション

def tokenize(text):
    """MeCabを使って単語トークンに分割"""
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


def ai_chat_bot(prompt):
   # vocab 辞書を反転させてインデックスからトークンに変換できるようにする
   reverse_vocab = {idx: token for token, idx in vocab.items()}  # tgt_vocabに変える

   # プロンプトを入力して推論
   output_sequence = generate_text(decoder, linear_layer, embedding_layer, vocab, prompt)

   # 出力トークンをトークンに変換
   output_tokens_as_text = [reverse_vocab[token_idx] for token_idx in output_sequence]

   # 結果を表示
   return "".join(output_tokens_as_text).replace("<end>", "")


# In[7]:


# AIチャットボットの送受信テスト
if __name__ == '__main__':
    print("AIチャットボットからの出力: ", ai_chat_bot("太郎は"))


# In[ ]:




