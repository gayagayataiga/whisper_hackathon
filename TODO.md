# TODO — B / C 残課題

初回コミット時に「A: 壊れている / 古い記述の修正」までは反映済み。
ここでは B (動作改善) と C (設定外出し) で未着手のものをまとめる。

## B. 動作・整合性まわり

### B1. `POST /reset_context` を追加
- **場所**: `Script/whisper_server.py`
- **現状**: `_previous_text` がモジュールグローバルとして直前の認識テキストを保持しており、`initial_prompt` に渡される。プロセス再起動以外でこれをクリアする手段がない。
- **やること**:
  - `POST /reset_context` を追加し、`_previous_text` を空文字に戻す。
  - レスポンスは `{"status": "ok"}` 程度で十分。
  - `interface.py` 側にも `POST /reset_context` を生やし、推論サーバーへ転送するか、もしくは Raspi から直接 8001 を叩く前提にするかを決める（受け子経由に統一する方が運用しやすい）。
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
  - `shutdown` で `del whisper_model` + `torch.cuda.empty_cache()` を入れておくと、`uvicorn --reload` 時に VRAM が溜まらない。

## C. 設定の外出し

### C1. `INFERENCE_URL` / `RASPI_URL` を環境変数で上書き可能に
- **場所**: `Script/interface.py:40-47`
- **現状**: ハードコード。Raspi の IP が変わるたびに `interface.py` を編集して再起動している (`docs/raspi_network.md` 参照)。
- **やること**:
  ```python
  import os
  INFERENCE_URL = os.environ.get("WHISPER_INFERENCE_URL", "http://localhost:8001/transcribe")
  RASPI_URL     = os.environ.get("WHISPER_RASPI_URL",     "http://10.27.72.53:9000/command")
  ```
- **`start.sh` 側**: `WHISPER_RASPI_URL=http://192.168.10.2:9000/command ./start.sh` で切り替えられるようにし、デフォルトは現行の無線 IP のまま。
- **`docs/raspi_network.md`**: 編集ではなく環境変数で切り替える運用に書き換える。
- **`status.sh:10` の `grep -E '^RASPI_URL' interface.py | sed ...`**: Python ソース文字列パースで脆い。環境変数化後はそれを表示するか、`/health` のレスポンスに `raspi_url` を含めて取得するのが望ましい。

### C2. `numpy<2.0` 制約の緩和
- **場所**: `requirements.txt:42`
- **現状**: コメントに「numpy 2.x は faster-whisper との互換性問題あり」とあり `numpy<2.0` で固定。
- **理由**: faster-whisper 1.1+ は numpy 2 を許容。ctranslate2 4.5+ も同様。
- **やること**:
  - 制約を `numpy>=1.21,<3.0` 程度に緩める。
  - 緩めた状態で `import faster_whisper; from faster_whisper import WhisperModel` がエラーなく動くことを確認。
  - 他に numpy を使っているのは `mock_raspi.py` (sin 波生成 / リサンプリング) と `Script/benchmark/*` 系のみ。numpy 2 でも互換のはず。

## 着手順の目安

1. C2 (numpy 制約緩和) — 1 行修正 + 動作確認のみ。最小コスト。
2. C1 (URL 環境変数化) — `interface.py` 数行 + `status.sh` 整理。中コスト。
3. B2 (lifespan 移行) — 2 ファイル書き換え。動作確認は `start.sh` で立てて疎通確認。
4. B1 (`/reset_context`) — エンドポイント追加。`mock_raspi.py` か `curl` でテスト可能。
