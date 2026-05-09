# Raspberry Pi ネットワーク設定メモ

受け子サーバー (`Script/interface.py`) が文字起こし結果を Raspi へ送信する際の URL は、
環境変数 `WHISPER_RASPI_URL` で指定する。`Script/interface.py` の編集は不要。

## IP アドレス一覧

| 接続方式 | IP アドレス     | ポート | 備考                     |
| -------- | --------------- | ------ | ------------------------ |
| 無線     | `10.27.72.53`   | 9000   | 現在使用中 (2026-05-02)  |
| 有線     | `192.168.10.2`  | 9000   | 有線接続に戻したとき用   |

Jetson 側 (有線) は `eno1` に `192.168.10.1/24` を固定。
NetworkManager プロファイル `Wired connection 1` に保存済み
(MAC `3C:6D:66:B1:AB:FA` で紐付け、autoconnect-priority=100)。

## 起動方法

`Script/start.sh` 起動時に `WHISPER_RASPI_URL` を環境変数で渡す:

```bash
# 無線接続
WHISPER_RASPI_URL=http://10.27.72.53:9000/command ./start.sh

# 有線接続
WHISPER_RASPI_URL=http://192.168.10.2:9000/command ./start.sh
```

`WHISPER_RASPI_URL` は必須。未設定の場合は `start.sh` がエラーで終了する
(IP の取り違えを防ぐためデフォルト値を持たせていない)。

毎回手で指定するのが煩雑なら、shell rc に export しておく:

```bash
# ~/.bashrc などに追記
export WHISPER_RASPI_URL=http://10.27.72.53:9000/command
```

## 推論サーバー URL の変更 (任意)

通常はローカル (`http://localhost:8001`) で固定。リモート推論サーバーを使う場合のみ
`WHISPER_INFERENCE_URL` を指定する (path は含めず base URL のみ):

```bash
WHISPER_INFERENCE_URL=http://other-host:8001 \
WHISPER_RASPI_URL=http://10.27.72.53:9000/command \
./start.sh
```

## 起動後の確認

`./status.sh` で現在の `WHISPER_RASPI_URL` を表示できる
(`Script/.env.runtime` に記録された起動時の値を参照)。
