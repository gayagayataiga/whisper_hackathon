# TODO — B / C 残課題

初回コミット時に「A: 壊れている / 古い記述の修正」までは反映済み。
ここでは B (動作改善) と C (設定外出し) で未着手のものをまとめる。

完了済み (詳細は `git log` 参照):
- C2 (numpy 制約緩和)
- B1 (`POST /reset_context` 追加 + `/health` の `previous_text_chars`)
- C1 (URL を環境変数 `WHISPER_RASPI_URL` / `WHISPER_INFERENCE_URL` に外出し)

残るのは下記 2 件のみで、いずれも **優先度低（当面やらない）**。

## 優先度低（当面やらない）

### B2. `@app.on_event("startup")` を `lifespan` に移行
- **動機**: FastAPI は `on_event` を deprecated 扱い。Starlette 0.36+ で警告が出る。将来的に削除される。
- **やらない理由**: 本番は 1 モデル固定で安定稼働しており、移行の実利が薄い。Starlette が実際に `on_event` を削除する段階で対応する。

- **場所**:
  - `Script/whisper_server.py:116` (`startup_event` でモデルロード)
  - `Script/interface.py:92` (`startup_event` でディレクトリ作成 + ログ)
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
- **動機**: モデル A → B → C と切り替える際、前のモデルの VRAM が解放されないまま次がロードされて、OOM や速度低下の原因になりうる。
- **やらない理由**: 本番では単一モデル運用のため不要。ベンチで OOM や速度劣化に実際にぶつかったときに対応する。

- **場所**: `Script/benchmark/*` 系（モデルを順番に差し替えてる箇所）
- **やること**: 切り替え処理に以下を挟む。
  ```python
  import gc, torch
  del whisper_model
  gc.collect()
  torch.cuda.empty_cache()  # ctranslate2 経由なので効果は限定的だが害はない
  ```
- **B2 との関係**: lifespan の shutdown フックはサーバー終了時にしか走らないので、ベンチの切り替えには使えない。別物として扱う。
