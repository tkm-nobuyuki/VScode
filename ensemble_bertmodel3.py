import os
import glob
import numpy
import pprint
import torch
import pandas as pd
from pathlib import Path
from functools import partial
from transformers import BertModel, BertConfig
import json

# 外部ファイル make_grid.output_data の代替関数 (ダミー)
def output_data(path, output_json, json_name=False):
   
    try:
        if json_name:
            with open(path, 'w') as f:
                json.dump(output_json, f, indent=4)
    except Exception as e:
        print(f"WARN: Could not save result file: {e}")

# ==== BERTモデルの定義 (変更なし) ====
class BertForTrajectoryPrediction(torch.nn.Module):
    def __init__(self, hidden_size, n_layers, num_attention_heads, window_size, step_count):
        super().__init__()
        self.step_count = step_count
        self.output_dim = 2
        bert_config = BertConfig(
            hidden_size=hidden_size, num_hidden_layers=n_layers,
            num_attention_heads=num_attention_heads, intermediate_size=hidden_size * 4,
            max_position_embeddings=window_size
        )
        self.embedding = torch.nn.Linear(self.output_dim, hidden_size)
        self.encoder = BertModel(bert_config)
        self.fc = torch.nn.Linear(hidden_size, self.output_dim * self.step_count)

    def forward(self, x):
        batch_size, _, _ = x.size()
        x_embedded = self.embedding(x)
        encoder_output = self.encoder(inputs_embeds=x_embedded)
        last_hidden_state = encoder_output.last_hidden_state[:, -1, :]
        prediction = self.fc(last_hidden_state)
        return prediction.view(batch_size, self.step_count, self.output_dim)

# ==== データ処理関数 (変更なし) ====
def read_and_grid_trajectory(file_path, lat_conf, lon_conf):
    """単一の軌跡ファイル(.csv)を読み込み、クリーニングとグリッド変換を行う"""
    try:
        names = ('Latitude', 'Longitude', 'All_0', 'Altitude', 'Date_num', 'Date', 'Time')
        df = pd.read_csv(file_path, sep=',', header=None, names=names, on_bad_lines='skip')
        df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
        
        lat_len = (lat_conf["GOAL"] - lat_conf["START"]) / lat_conf["GRID"]
        lon_len = (lon_conf["GOAL"] - lon_conf["START"]) / lon_conf["GRID"]
        
        df['lat_grid'] = ((df['Latitude'] - lat_conf["START"]) / lat_len).astype(int)
        df['lon_grid'] = ((df['Longitude'] - lon_conf["START"]) / lon_len).astype(int)
        coords = df[['lat_grid', 'lon_grid']].values
        
        valid_mask = (coords[:, 0] >= 0) & (coords[:, 0] < lat_conf["GRID"]) & \
                     (coords[:, 1] >= 0) & (coords[:, 1] < lon_conf["GRID"])
        
        return coords[valid_mask]
    except Exception:
        return numpy.array([])

# ==== モデルの設定 (CROWD_COUNT=10) ====
MODEL_DATE = "20251119_012724" 
TEST_SIZE = 0.2 
USER = list(range(0, 182))
CROWD_COUNT = 10 
IS_SAVE_RESULT = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BERT_MODEL_CONFIG = None

def get_bert_config(model_date):
    """summary.jsonから学習時の設定をロードする"""
    global BERT_MODEL_CONFIG
    summary_path = os.path.join("output_folder", "Model", "BERT_Optimized", model_date, "summary.json")
    
    # 外部ファイルがない場合のデフォルト設定
    class Config:
        WINDOW_SIZE = 100 
        STEP_COUNT = 10 
        HIDDEN_SIZE = 64 
        N_LAYERS = 1 
        NUM_ATTENTION_HEADS = 4
        WIDTH_LAT = {"START": 39.75, "GOAL": 40.15, "GRID": 80}
        WIDTH_LON = {"START": 116.15, "GOAL": 116.55, "GRID": 80}
        TEST_SIZE = 0.2
    
    config_data = {k: v for k, v in Config.__dict__.items() if not k.startswith('__')}
    
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                loaded_config = json.load(f).get("config", {})
            if loaded_config:
                config_data.update(loaded_config)
        except Exception:
            pass # ロード失敗時はデフォルト設定を使用
        
    BERT_MODEL_CONFIG = {
        "WINDOW_SIZE": config_data["WINDOW_SIZE"],
        "STEP_COUNT": config_data["STEP_COUNT"],
        "HIDDEN_SIZE": config_data["HIDDEN_SIZE"],
        "N_LAYERS": config_data["N_LAYERS"],
        "NUM_ATTENTION_HEADS": config_data["NUM_ATTENTION_HEADS"],
        "WIDTH_LAT": config_data["WIDTH_LAT"],
        "WIDTH_LON": config_data["WIDTH_LON"],
        "TEST_SIZE": config_data.get("TEST_SIZE", TEST_SIZE),
    }
    return BERT_MODEL_CONFIG

# -----------------------------------------------------------

def main():
    global BERT_MODEL_CONFIG
    
    try:
        BERT_MODEL_CONFIG = get_bert_config(MODEL_DATE)
    except Exception as e:
        print(f"FATAL: BERT Config Loading Failed: {e}"); return
        
    all_json_data = get_test_data_bert(MODEL_DATE, USER) 
    folder_path = all_json_data["folder_path"]
    crowd_model = set_crowd_model() 
    count_result = []
    
    print(f"--- BERT Ensemble Evaluation Start (Crowd Count={CROWD_COUNT}, Deduplication Enabled, Sliding Window Test + Batching) ---")

    for user_data in all_json_data["user_data"]:
        setting_json = user_data["user_setting"]
        user_id_num = setting_json['user']
        user_id_str = f"{user_id_num:03}"
        
        # ---------------------------------------------------
        # ▼ Private Model Prediction (BERT)
        li_private_goals_10_steps, test_inputs, test_goals = get_private_model_bert(user_id_num, user_id_str, MODEL_DATE)
        
        if not li_private_goals_10_steps:
             continue
        
        # ▼ Crowd Model Prediction
        # テストデータの最後の点（t=100）を入力としてCrowd予測を取得
        last_input_points = [tuple(arr[-1]) for arr in test_inputs]
        li_crowd_goal = get_crowd_model(last_input_points, crowd_model)
        # ---------------------------------------------------
        
        crowd_count = 0
        private_count = 0
        ensemble_count = 0
        total_predictions = len(test_goals)
        
        # BERT予測経路とCrowd候補、正解を同時に処理
        for bert_goals, crowd_goal_item, ans_goal_10_steps in zip(li_private_goals_10_steps, li_crowd_goal, test_goals):
            
            # 評価の正解は、予測シーケンスの**最後の点 (t+STEP_COUNT)** とする
            ans_goal = tuple(ans_goal_10_steps[-1]) 
            
            # --- Crowdの候補リストを作成 ---
            crowd_goal = list(crowd_goal_item.keys()) 
            
            # --- Private (BERT) の候補リストを作成（10点すべてを使用） ---
            private_goal = [tuple(p) for p in bert_goals] 
            
            # --- Ensembleの候補リストを作成 ---
            ensemble_goal = list(set(private_goal + crowd_goal)) 
            
            # --- 正解判定 (In-List Accuracy) ---
            crowd_count += 1 if ans_goal in crowd_goal else 0
            private_count += 1 if ans_goal in private_goal else 0 
            ensemble_count += 1 if ans_goal in ensemble_goal else 0 
        
        if total_predictions > 0:
            count_result.append({
                "user":user_id_num, 
                "crowd_count":crowd_count, 
                "private_count":private_count, 
                "ensemble_count":ensemble_count, 
                "total":total_predictions,
                "crowd_answer_rate":crowd_count/total_predictions, 
                "private_answer_rate":private_count/total_predictions, 
                "ensemble_answer_rate":ensemble_count/total_predictions
            })

    ## 結果集計と表示 
    if not count_result:
        print("No results processed.")
        return
        
    crowd_count_sum = sum(r['crowd_count'] for r in count_result)
    private_count_sum = sum(r['private_count'] for r in count_result)
    ensemble_count_sum = sum(r['ensemble_count'] for r in count_result)
    total = sum(r['total'] for r in count_result)
    
    crowd_answer_rate_sum = sum(r['crowd_answer_rate'] for r in count_result)
    private_answer_rate_sum = sum(r['private_answer_rate'] for r in count_result)
    ensemble_answer_rate_sum = sum(r['ensemble_answer_rate'] for r in count_result)
    
    crowd_all_answer_rate = crowd_count_sum / total if total > 0 else 0
    private_all_answer_rate = private_count_sum / total if total > 0 else 0
    ensemble_all_answer_rate = ensemble_count_sum / total if total > 0 else 0
    
    crowd_answer_rate_average = crowd_answer_rate_sum / len(count_result)
    private_answer_rate_average = private_answer_rate_sum / len(count_result)
    ensemble_answer_rate_average = ensemble_answer_rate_sum / len(count_result)
    
    crowd_count_zero = sum(1 for r in count_result if r['crowd_count'] == 0)
    not_crowd_count_zero = len(count_result) - crowd_count_zero
    
    print("\n" + "="*40)
    print(" 全体集計結果 (BERT Private Model)")
    print("予測したユーザー数:", len(count_result))
    print("crowdの正解率が0%のユーザー数:", crowd_count_zero)
    print("crowdの正解率が0%でないユーザー数:", not_crowd_count_zero)
    print("全ユーザーの正解数の合計（crowd）:", crowd_count_sum)
    print("全ユーザーの正解数の合計（private）:", private_count_sum)
    print("全ユーザーの正解数の合計（ensemble）:", ensemble_count_sum)
    print("予測の総数:", total)
    print(f"全ユーザーの正解率（crowd）:{crowd_all_answer_rate:.2%}")
    print(f"全ユーザーの正解率（private）:{private_all_answer_rate:.2%}")
    print(f"全ユーザーの正解率（ensemble）:{ensemble_all_answer_rate:.2%}")
    print(f"全ユーザーの正解率の平均（crowd）:{crowd_answer_rate_average:.2%}")
    print(f"全ユーザーの正解率の平均（private）:{private_answer_rate_average:.2%}")
    print(f"全ユーザーの正解率の平均（ensemble）:{ensemble_answer_rate_average:.2%}")
    print("="*40)

    final_summary = {
        "all_user_count": len(count_result),
        "crowd_count_zero_user_count": crowd_count_zero,
        "not_crowd_count_zero_user_count": not_crowd_count_zero,
        "crowd_count_sum": crowd_count_sum,
        "private_count_sum": private_count_sum,
        "ensemble_count_sum": ensemble_count_sum,
        "total": total,
        "crowd_all_answer_rate": crowd_all_answer_rate,
        "private_all_answer_rate": private_all_answer_rate,
        "ensemble_all_answer_rate": ensemble_all_answer_rate,
        "crowd_answer_rate_average": crowd_answer_rate_average,
        "private_answer_rate_average": private_answer_rate_average,
        "ensemble_answer_rate_average": ensemble_answer_rate_average
    }
    count_result.append(final_summary)
    
    if IS_SAVE_RESULT:
        # ファイル名を変更し、Sliding Windowの結果であることを示す
        output_data(os.path.join(folder_path, "ensemble_bert_result_crowd10_sliding_test_batched.json"), output_json=count_result, json_name=True)

# -----------------------------------------------------------
# ▼ データ取得関数 (変更なし)
# -----------------------------------------------------------
def get_test_data_bert(model_date, li_user:list):
    """BERT学習時の設定をロードし、ユーザーリストを準備する"""
    base_folder = os.path.join("output_folder", "Model", "BERT_Optimized", model_date)
    folder_path = base_folder
    li_user_data = []

    for user_id_num in li_user:
        user_id_str = f"{user_id_num:03}"
        if not os.path.isdir(os.path.join(base_folder, user_id_str)): continue
        user_setting = BERT_MODEL_CONFIG.copy()
        user_setting["user"] = user_id_num
        li_user_data.append({"user_input": {"inputs": user_id_num, "goal": user_id_num}, "user_setting": user_setting})
        
    output_json = {"user_data":li_user_data, "folder_path":folder_path}
    return output_json


# -----------------------------------------------------------
# ▼ Crowd Model 取得関数 (変更なし)
# -----------------------------------------------------------
def set_crowd_model():
    crowd_path = os.path.join("output_folder", "Model", "Crowd")
    crowd_model_file = os.path.join(crowd_path, "train_model.npy")
    
    if not os.path.exists(crowd_model_file):
        return {} 
        
    try:
        crowd_model = numpy.load(crowd_model_file, allow_pickle=True).item()
    except Exception:
        return {}
    
    dict_total = {key: sum(crowd_model[key].values()) for key in crowd_model}
    crowd_model = {key: {key2: round(crowd_model[key][key2] / dict_total[key] * 100, 2) for key2 in crowd_model[key]} for key in crowd_model if dict_total[key] > 0}
    crowd_model = {key: sorted(crowd_model[key].items(), key=lambda x: x[1], reverse=True)[:CROWD_COUNT] for key in crowd_model}
    crowd_model = {key: {key2: value2 for key2, value2 in crowd_model[key]} for key in crowd_model}
    return crowd_model

def get_crowd_model(li_test_data_last_points, crowd_model):
    """Private予測に使用した入力シーケンスの最後の点を受け取り、Crowd予測を返す"""
    li_crowd_goal = list(map(lambda x: crowd_model.get(x, {}), li_test_data_last_points))
    return li_crowd_goal

# -----------------------------------------------------------
# ▼ Private Model 予測関数 (スライディングウィンドウ + バッチ処理を導入)
# -----------------------------------------------------------
def get_private_model_bert(user_id_num, user_id_str, model_date):
    """
    BERTモデルを使用してテストデータに対する予測を実行する。
    スライディングウィンドウで大量サンプルを生成し、バッチ処理でメモリ不足を回避。
    """
    config = BERT_MODEL_CONFIG
    
    model_path = os.path.join("output_folder", "Model", "BERT_Optimized", model_date, user_id_str, "model.pth")
    if not os.path.exists(model_path): return [], [], []
        
    try:
        model = BertForTrajectoryPrediction(
            hidden_size=config["HIDDEN_SIZE"], n_layers=config["N_LAYERS"],
            num_attention_heads=config["NUM_ATTENTION_HEADS"],
            window_size=config["WINDOW_SIZE"], step_count=config["STEP_COUNT"]
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        model.eval()
    except Exception:
        return [], [], []
        
    # --- データの準備 (スライディングウィンドウ) ---
    data_folder = os.path.join("output_folder", "Data", user_id_str)
    trajectory_files = sorted(glob.glob(os.path.join(data_folder, "Trajectory", "*.csv")))
    
    num_test_files = int(len(trajectory_files) * config["TEST_SIZE"])
    if len(trajectory_files) > 1 and num_test_files == 0: num_test_files = 1
    
    test_files = trajectory_files[-num_test_files:]
    
    read_func = partial(read_and_grid_trajectory, 
                        lat_conf=config['WIDTH_LAT'], 
                        lon_conf=config['WIDTH_LON'])
                        
    all_inputs, all_goals = [], []
    window_size = config["WINDOW_SIZE"]
    step_count = config["STEP_COUNT"]
    
    SLIDING_STEP = 5 

    for traj_file in test_files:
        trajectory = read_func(traj_file)
        full_length = len(trajectory)
        
        if full_length >= window_size + step_count:
            max_start_index = full_length - window_size - step_count + 1 
            
            for start_index in range(0, max_start_index, SLIDING_STEP):
                end_index_input = start_index + window_size
                end_index_goal = end_index_input + step_count
                
                input_data = trajectory[start_index:end_index_input]
                goal_data = trajectory[end_index_input:end_index_goal]
                
                all_inputs.append(input_data)
                all_goals.append(goal_data)

    if not all_inputs: return [], [], []
    
    # inputsをテンソルに変換 (全データを一度にロードするが、すぐにバッチに分ける)
    inputs_tensor = torch.tensor(numpy.array(all_inputs), dtype=torch.float32)
    
    # --- 予測実行とバッチ処理 ---
    
    #  修正箇所: BATCH_SIZEを設定し、予測をバッチで実行
    BATCH_SIZE = 256 
    all_preds_list = []

    with torch.no_grad():
        num_batches = (len(inputs_tensor) + BATCH_SIZE - 1) // BATCH_SIZE
        
        # 非正規化に必要なグリッドサイズ
        lat_size = config["WIDTH_LAT"]["GRID"] + 1
        lon_size = config["WIDTH_LON"]["GRID"] + 1

        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(inputs_tensor))
            
            # バッチをDEVICE (CPUまたはGPU) に送る
            batch_inputs = inputs_tensor[start_idx:end_idx].to(DEVICE)

            # バッチごとにモデルに渡す
            batch_outputs = model(batch_inputs)
            
            # 非正規化 (テンソルのまま実行)
            batch_outputs[:, :, 0] *= lat_size
            batch_outputs[:, :, 1] *= lon_size
            
            # グリッド座標に丸め、CPUに戻してNumPyに変換
            batch_preds = torch.round(batch_outputs).cpu().numpy()
            
            all_preds_list.append(batch_preds)

    # 修正箇所: 全ての予測結果を結合
    preds = numpy.concatenate(all_preds_list, axis=0)

    # 予測結果をリスト化
    output_list_of_10_steps = [p.tolist() for p in preds]
    
    return output_list_of_10_steps, numpy.array(all_inputs), numpy.array(all_goals) 

if __name__ == '__main__':
    main()