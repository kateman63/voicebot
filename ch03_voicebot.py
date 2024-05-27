import base64
import os
from datetime import datetime

import numpy as np
import openai
import streamlit as st
from audiorecorder import audiorecorder
from gtts import gTTS


#### 기능 함수 구현 ####
def STT(audio):
    # 파일 저장
    filename = "input.mp3"
    wav_file = open(filename, "wb")
    wav_file.write(audio.tobytes())
    wav_file.close()

    # 음원 파일 열기
    audio_file = open(filename,"rb")
    # Whisper 모델을 활용해 텍스트 얻기
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    audio_file.close()

    # 파일 삭제
    os.remove(filename)
    return transcript["text"]

def ask_gpt(prompt, model):
    response = openai.ChatCompletion.create(model=model, message=prompt)
    system_message = response["choice"][0]["message"]
    return system_message["content"]

def TTS(response):
    # gTTS를 활용하여 음성 파일 생성
    filename = "output.mp3"
    tts = gTTS(text=response, lang="ko")
    tts.save(filename)

    # 음원 파일 자동 재생
    with open(filename, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="True">
            <source src="data:audio/mp3;base64,{64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True,)

    # 파일 삭제
    os.remove(filename)

#### 메인 함수 ####
def main():
    # 기본 설정
    st.set_page_config(
        page_title="음성 비서 프로그램",
        layout="wide")

    flag_start = False

    # 제목
    st.header("음성 비서 프로그램")

    # 구분선
    st.markdown("---")

    # 기본 설명
    with st.expander("음성 비서 프로그램에 관하여", expanded=True):
        st.write(
            """
             - 음성 비서 프로그램의 UI는 스트림릿을 활용했습니다.
             - STT(Speech-To-Text)는 OpenAI의 Whisper AI를 활용했습니다.
             - 답변은 OpenAI의 GPT 모델을 활용했습니다.
             - TTS(Text-To-Speech)는 구글의 Google Translate TTS를 활용했습니다.
             """
        )

        st.markdown("")

    # session state 초기화
    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    if "message" not in st.session_state:
        st.session_state["message"] = [{"role": "system", "content": "You are a thoughtful assistant. Respond to all input in 25 words and answer in korea"}]

    if "check_audio" not in st.session_state:
        st.session_state["check_audio"] = []

    # 사이드바 생성
    with st.sidebar:
        # Open AI API 키 입력받기
        openai.api_key = st.text_input(label="OPENAI API 키", placeholder="Enter Your API Key", value="", type="password")

        st.markdown("---")

        # GPT 모델을 선택하기 위한 라디오 버튼 생성
        model = st.radio(label="GPT 모델", options=["gpt-4", "gpt-3.5-turbo"])

        st.markdown("---")

        # 리셋 버튼 생성
        if st.button(label="초기화"):
            # 리셋 코드
            st.session_state["chat"] = []
            st.session_state[ "message" ] = [ {"role": "system",
                                               "content": "You are a thoughtful assistant. Respond to all input in 25 words and answer in korea"} ]

    # 기능 구현 공간
    col1, col2 = st.columns(2)
    with col1:
        # 왼쪽 영역 작성
        st.subheader("질문하기")
        # 음성 녹음 아이콘 추가
        audio = audiorecorder("클릭하여 녹음하기","녹음 중...")
        if len(audio) > 0 and not np.array_equal(audio, st.session_state["check_audio"]): # 녹음을 실행하면
            # 음성 재생
            st.audio(audio.tobytes())
            # 음원 파일에서 텍스트 추출
            question = STT(audio)

            # 채팅을 시각화하기 위한 질문 내용 저장
            now = datetime.now().strftime("%H:%M")
            st.session_state["chat"] = st.session_state["chat"] + [("user", now, question)]
            # 질문 내용 저장
            st.session_state["message"] = st.session_state["message"] + [{"role": "user", "content":  question}]
            # audio 버퍼를 확인하기 위해 오디오 정보 저장
            st.session_state["check_audio"] = audio
            flag_start = True

    with col2:
        # 오른쪽 영역 작성
        st.subheader("질문/답변")
        if flag_start:
            # ChatGPT에게 답변 얻기
            response = ask_gpt(st.session_state["message"], model)

            #GPT 모델에 넣을 프롬프트를 위해 답변 내용 저장
            st.session_state["message"] = st.session_state["message"] + [{"role": "system", "content" : response}]

            # 채팅 시각화를 위한 답변 내용 저장
            now = datetime.now().strftime("%H:%M")
            st.session_state["chat"] = st.session_state["chat"] + [{"bot", now, response}]

            # 채팅 형식으로 시각화
            for sender, time, message in st.session_state["chat"]:
                if sender == "user":
                    st.write(f'<div style="display:flex;align-items:center;"><div style="background-color:#007AFF;color:white;border-radius:12px;padding:8px 12px;margin-right:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)
                    st.write()
                else:
                    st.write(f'<div style="display:flex;align-items:center;justify-12px;flex-end"><div style="background-color:lightgray;color:white;border-radius:12px;padding:8px 12px;margin-left:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)
                    st.write("")

            # gTTS를 활용하여 음성 파일 생성 및 재생
            TTS(response)


if __name__ == "__main__":
        main()