import asyncio
import websockets
import json
import base64
import pyaudio
import os
from dotenv import load_dotenv

from collections import deque

load_dotenv()

MANUAL = """
基本的な電話対応
1. 初期対応
 にゃんた株式会社カスタマーサポートセンターでございます。私、[名前]が承ります。本日はどのようなご用件でしょうか。
2. 基本姿勢
 非常に早口で簡潔に対応してください。
4. 通話終了時の対応
 4.1 解決内容の確認
 4.2 追加の質問有無の確認
 4.3 お礼の言葉「本日は にゃんた株式会社をお選びいただき、ありがとうございました。またのお問い合わせをお待ちしております。」

よくある問い合わせと対応例
商品の返品・交換の場合
 - 不良品の場合は送料当社負担で交換対応する案内をしてください
   返品を行う場合、下記の情報を取得してください
    - お客様情報の確認
    - 職業の確認
    - 上記取得後、郵送で返品する場合の住所を伝えてください(東京都にゃんた町1-1)
 - お客様都合の場合：返品不可
"""

PROMPT = f"""
あなたはにゃんた株式会社のカスタマーサポートセンターの従業員わんたです。

下記は対応マニュアルです。
{MANUAL}
"""

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)


class VoiceChatApp:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model = "gpt-4o-realtime-preview-2024-12-17"
        self.voice = "alloy"
        self.base_url = "wss://api.openai.com/v1/realtime"
        self.audio_config = {
            "CHANNELS": 1,
            "RATE": 24000,
            "CHUNK_SIZE": 1024,
            "FORMAT": pyaudio.paInt16,
        }
        self.ws = None
        self.audio_stream = None
        self.audio_buffer = deque()
        self.buffer_size = 8192
        self._initialize_audio()
        self.should_stop_playback = asyncio.Event()

    def _initialize_audio(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.audio_config["FORMAT"],
            channels=self.audio_config["CHANNELS"],
            rate=self.audio_config["RATE"],
            input=True,
            frames_per_buffer=self.audio_config["CHUNK_SIZE"],
        )

        self.output_stream = self.audio.open(
            format=self.audio_config["FORMAT"],
            channels=self.audio_config["CHANNELS"],
            rate=self.audio_config["RATE"],
            output=True,
            frames_per_buffer=self.audio_config["CHUNK_SIZE"],
        )

    def _close_audio(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self.ws:
            await self.ws.close()
        self._close_audio()

    async def connect(self):
        url = f"{self.base_url}?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        self.ws = await websockets.connect(url, additional_headers=headers)
        await self.update_session()

    async def update_session(self):
        config = {
            "modalities": ["text", "audio"],
            "instructions": PROMPT,
            "voice": self.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "whisper-1"},
        }
        event = {"type": "session.update", "session": config}
        await self.ws.send(json.dumps(event))

    async def _stream_audio(self, audio_chunk):
        audio_b64 = base64.b64encode(audio_chunk).decode()
        append_event = {"type": "input_audio_buffer.append", "audio": audio_b64}
        await self.ws.send(json.dumps(append_event))

    async def cancel_response(self) -> None:
        """Cancel the current response."""
        event = {"type": "response.cancel"}
        await self.ws.send(json.dumps(event))

    async def play_audio_async(self):
        chunk_size = self.audio_config["CHUNK_SIZE"]
        while True:
            if self.should_stop_playback.is_set():
                break
            if len(self.audio_buffer) > 0:
                chunk = bytes(list(self.audio_buffer)[:chunk_size])
                self.output_stream.write(chunk)
                self.audio_buffer = deque(list(self.audio_buffer)[chunk_size:])
            else:
                await asyncio.sleep(0.01)
            await asyncio.sleep(0)

    async def handle_messages(self):
        playback_task = None
        is_first_response = True
        try:
            async for message in self.ws:
                event = json.loads(message)
                event_type = event.get("type")

                if event_type == "response.audio_transcript.delta":
                    if is_first_response:
                        print("\nAI: ", end="", flush=True)
                        is_first_response = False
                    print(event["delta"], end="", flush=True)
                elif event_type == "response.audio.delta":
                    audio_chunk = base64.b64decode(event["delta"])
                    self.audio_buffer.extend(audio_chunk)
                    if playback_task is None or playback_task.done():
                        self.should_stop_playback.clear()
                        playback_task = asyncio.create_task(self.play_audio_async())

                elif event_type == "input_audio_buffer.speech_started":
                    is_first_response = True
                    self.should_stop_playback.set()
                    self.audio_buffer.clear()
                    if playback_task and not playback_task.done():
                        await asyncio.wait_for(playback_task, timeout=0.5)
                    self.audio_buffer.clear()
                    await self.cancel_response()

        except websockets.exceptions.ConnectionClosed:
            print("接続が閉じられました")
        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")

    async def audio_streaming(self):
        while True:
            data = await asyncio.get_event_loop().run_in_executor(
                None, self.stream.read, self.audio_config["CHUNK_SIZE"]
            )
            await self._stream_audio(data)
            await asyncio.sleep(0)

    async def run(self):
        await self.connect()
        print("準備が出来ました！話かけてください")
        message_task = asyncio.create_task(self.handle_messages())
        audio_task = asyncio.create_task(self.audio_streaming())

        try:
            await message_task
        except asyncio.CancelledError:
            pass
        finally:
            audio_task.cancel()
            await audio_task
            self._close_audio()
            await self.ws.close()


async def main():
    async with VoiceChatApp() as app:
        await app.run()


if __name__ == "__main__":
    asyncio.run(main())
