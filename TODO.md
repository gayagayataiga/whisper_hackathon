# TODO — B / C 残課題

初回コミット時に「A: 壊れている / 古い記述の修正」までは反映済み。
ここでは B (動作改善) と C (設定外出し) で未着手のものをまとめる。

## B. 動作・整合性まわり

### B1. `POST /reset_context` を追加
- **前提**: 本サーバーは uvicorn workers=1 で運用する（`_previous_text` のプロセス間共有はしない）。
  `start.sh` でも `--workers 1` を明示するか、デフォルト（=1）に依存していることをコメントで残す。

- **場所**: `Script/whisper_server.py`
- **現状**: `_previous_text` がモジュールグローバルとして直前の認識テキストを保持しており、`initial_prompt` に渡される。プロセス再起動以外でこれをクリアする手段がない。
- **やること**:
  - `POST /reset_context` を追加し、`_previous_text` を空文字に戻す。
  - レスポンスは `{"status": "ok"}` 程度で十分。
  - `interface.py` 側にも `POST /reset_context` を生やし、推論サーバーへ転送するか、もしくは Raspi から直接 8001 を叩く前提にするかを決める（受け子経由に統一する方が運用しやすい）。interface 経由にする。

- **テスト観点**: 1 回 `/transcribe` した後 `/reset_context` → `_previous_text` が空であることを `/health` か返り値で確認できると良い（`/health` に `previous_text_chars` を生やすのが一案）。

### B2. `@app.on_event("startup")` を `lifespan` に移行
- **場所**:
  - `Script/whisper_server.py:115` (`startup_event` でモデルロード)
  - `Script/interface.py:81` (`startup_event` でディレクトリ作成 + ログ)
- **理由**: FastAPI は `on_event` を deprecated 扱い。Starlette 0.36+ で警告が出る。将来的に削除される。
- **やること**:
  ```python
  from contextlib import asynccontextmanager

  @asynccontextmanager
  async def lifespan(app: FastAPI):
      # 旧 startup_event の中身
      yield
      # 必要なら shutdown 処理（モデル破棄など）

  app = FastAPI(lifespan=lifespan, title=..., version=...)
  ```
- **注意点**:
  - `whisper_model` を `app.state.whisper_model` に持たせるとテスト時の差し替えが楽になるが、現状のグローバル変数のままでも動く。スコープ膨らませるならグローバル維持で OK。
- `shutdown` での VRAM 解放はベンチマーク（モデル切り替え）の方が重要なので、別タスクとして切り出す（下記 B3 参照）。lifespan 移行自体には含めない。

### B3. ベンチマーク時のモデル切り替えで VRAM を明示解放
- **場所**: `Script/benchmark/*` 系（モデルを順番に差し替えてる箇所）
- **現状**: モデル A → B → C と切り替える際、前のモデルの VRAM が解放されないまま次がロードされて、OOM や速度低下の原因になりうる。
- **やること**: 切り替え処理に以下を挟む。
```python
  import gc, torch
  del whisper_model
  gc.collect()
  torch.cuda.empty_cache()  # ctranslate2 経由なので効果は限定的だが害はない
```
- **B2 との関係**: lifespan の shutdown フックはサーバー終了時にしか走らないので、ベンチの切り替えには使えない。別物として扱う。

## C. 設定の外出し

### C1. `INFERENCE_URL` / `RASPI_URL` を環境変数で上書き可能に
- **場所**: `Script/interface.py:40-47`
- **現状**: ハードコード。Raspi の IP が変わるたびに `interface.py` を編集して再起動している (`docs/raspi_network.md` 参照)。
- **やること**:
  - `interface.py`:
```python
    import os
    INFERENCE_URL = os.environ.get("WHISPER_INFERENCE_URL", "http://localhost:8001/transcribe")
    RASPI_URL     = os.environ["WHISPER_RASPI_URL"]  # 必須。未設定なら KeyError で起動失敗
```
  - `start.sh`: 起動時に必須チェック + ランタイムファイル出力。
```bash
    : "${WHISPER_RASPI_URL:?WHISPER_RASPI_URL is required}"
    echo "WHISPER_RASPI_URL=$WHISPER_RASPI_URL" >  .env.runtime
    echo "INFERENCE_URL=${WHISPER_INFERENCE_URL:-http://localhost:8001/transcribe}" >> .env.runtime
```
  - `status.sh`: `.env.runtime` を読む（Python ソースの grep をやめる）。
```bash
    source .env.runtime
    echo "RASPI_URL: $WHISPER_RASPI_URL"
```
- **`docs/raspi_network.md`**: 編集ではなく環境変数で切り替える運用に書き換える。
- **設計判断メモ**: `/config` エンドポイントを切る案もあったが、修正箇所が 3 ファイルに増える上、env と実値のズレを防ぐ仕組みが別途必要になるため不採用。`.env.runtime` 経由なら start.sh の 1 箇所で同期が取れる。


### C2. `numpy<2.0` 制約の緩和
- **場所**: `requirements.txt:42`
- **現状**: コメントに「numpy 2.x は faster-whisper との互換性問題あり」とあり `numpy<2.0` で固定。
- **理由**: faster-whisper 1.1+ は numpy 2 を許容。ctranslate2 4.5+ も同様。
- **やること**:
  - 制約を `numpy>=1.21,<3.0` 程度に緩める。
  - 緩めた状態で `import faster_whisper; from faster_whisper import WhisperModel` がエラーなく動くことを確認。
- 動作確認手順:
    1. `requirements.txt` 編集後、`pip install -r requirements.txt --upgrade`
    2. バージョン確認:
```bash
       python -c "import numpy; print('numpy:', numpy.__version__)"
       python -c "import ctranslate2; print('ctranslate2:', ctranslate2.__version__)"
       python -c "from faster_whisper import WhisperModel; print('import OK')"
```
    3. `ctranslate2` が 4.5 未満なら `requirements.txt` に `ctranslate2>=4.5` を追加。
    4. `./start.sh` 起動 → `curl -X POST http://localhost:8001/transcribe -F "file=@短い音声.wav"`
    5. **返ってきた日本語テキストの中身が壊れていないこと**を確認（200 が返るだけでは不十分。numpy 2 の dtype 互換問題は実行時に黙って異常値を出すパターンがある）。

## 優先度低（当面やらない）

### B2. lifespan 移行
- 動機は deprecated 警告のみ。本番は 1 モデル固定で安定稼働しているため、移行の実利が薄い。
- Starlette が実際に `on_event` を削除する段階で対応する。

### B3. ベンチマーク時の VRAM 明示解放
- 本番では単一モデル運用のため不要。
- ベンチで OOM や速度劣化に実際にぶつかったときに対応する。

## 着手順の目安

1. C2 (numpy 制約緩和)
2. B1 (`/reset_context`)
3. C1 (URL 環境変数化)