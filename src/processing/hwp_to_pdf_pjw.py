# hwp 파일을 pdf로 변환

import os
import win32com.client as win32

def convert_hwp_to_pdf(target_folder):
    try:
        hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")
        hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
    except Exception as e:
        print("오류: 한글 프로그램이 설치되어 있지 않거나 실행할 수 없습니다.")
        print(e)
        return

    output_folder = os.path.join(target_folder, "converted_pdfs")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"[변환 대상 폴더]: {target_folder}")
    print(f"[저장될 폴더]: {output_folder}")

    for root, dirs, files in os.walk(target_path):
        for filename in files:
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext == '.hwp':
                hwp_path = os.path.join(root, filename)
                # PDF 파일명 생성 (원본명.pdf)
                pdf_filename = f"{os.path.splitext(filename)[0]}.pdf"
                pdf_path = os.path.join(output_folder, pdf_filename)

                try:
                    hwp.Open(os.path.abspath(hwp_path))
                    
                    # PDF로 저장 설정
                    hwp.HAction.GetDefault("FileSaveAs_S", hwp.HParameterSet.HFileOpenSave.HSet)
                    hwp.HParameterSet.HFileOpenSave.filename = os.path.abspath(pdf_path)
                    hwp.HParameterSet.HFileOpenSave.Format = "PDF"
                    hwp.HAction.Execute("FileSaveAs_S", hwp.HParameterSet.HFileOpenSave.HSet)
                    
                    print(f"✅ 변환 완료: {pdf_filename}")
                    
                except Exception as e:
                    print(f"❌ 변환 실패: {filename}")
                    print(e)

                hwp.Clear(1)

    hwp.Quit()
    print("모든 변환 작업이 완료되었습니다.")

# 위치는 개인에 맞게 수정
target_path = r"C:\Users\USER\Desktop\AI중급프로젝트\중급_데이터\files"

# 함수 실행
convert_hwp_to_pdf(target_path)