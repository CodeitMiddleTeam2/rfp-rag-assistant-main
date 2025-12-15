import win32com.client as win32
import os
import glob

RAW_DIR = r"C:\Users\SAMSUNG\Documents\ai_agent\data\raw"

def convert_hwp_to_pdf(input_path, output_path):
    # 한글 실행
    hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")

    # 외부 프로그램 접근 허용을 위해 보안 설정 해제
    hwp.RegisterModule("FilePathCheckDLL", "SecurityModule")

    # HWP 열기
    hwp.Open(input_path, "HWP", "forceopen:true")

    # PDF 저장
    hwp.SaveAs(output_path, "PDF")

    # 종료
    hwp.Quit()


def batch_convert_raw_folder():
    # raw 폴더 내 모든 hwp 파일 수집
    hwp_files = glob.glob(os.path.join(RAW_DIR, "*.hwp"))

    if not hwp_files:
        print("[INFO] 변환할 HWP 파일이 없습니다.")
        return

    total = len(hwp_files)
    print(f"[INFO] 변환 대상 파일 수: {total}개")
    print("-" * 40)

    for idx, hwp_file in enumerate(hwp_files, start=1):
        # 출력 파일명: 같은 디렉토리, 확장자만 PDF로 변경
        pdf_file = os.path.splitext(hwp_file)[0] + ".pdf"

        print(f"[{idx}/{total}] 변환 중 → {os.path.basename(hwp_file)}")

        try:
            convert_hwp_to_pdf(hwp_file, pdf_file)
            print(f"   ✔ 완료 → {os.path.basename(pdf_file)}")
        except Exception as e:
            print(f"   ✖ 오류 발생: {e}")

    print("-" * 40)
    print("[SUCCESS] 모든 HWP → PDF 변환 완료!")


if __name__ == "__main__":
    batch_convert_raw_folder()
