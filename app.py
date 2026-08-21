import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN TRANG WEB
# ==========================================
st.set_page_config(page_title="CDSS Atrophic Gastritic - Hệ thống hỗ trợ ra quyết định lâm sàng Viêm Teo Dạ Dày", layout="wide")
st.title("🩺 Hệ thống Hỗ trợ Chẩn đoán Viêm Teo Dạ Dày và Hội chứng Y học cổ truyền")
st.markdown("*Clinical Decision Support System for Atrophic Gastritis and Traditional Medicine Syndromes*")
st.markdown("---")

# ==========================================
# 2. TỪ ĐIỂN MAP TÊN BIẾN & GOM NHÓM
# ==========================================
FEATURE_DICT = {
    'Dem_Age': {'group': 'Dịch tễ và Tiền sử (Epidemiology and Medical history)', 'label': 'Tuổi (Age)', 'type': 'number'},
    'Dem_Occupation_Office_worker': {'group': 'Dịch tễ và Tiền sử (Epidemiology and Medical history)', 'label': 'Nhân viên văn phòng (Office worker)', 'type': 'bool'},
    'Med_H_pylori_Status_Eradicated': {'group': 'Dịch tễ và Tiền sử (Epidemiology and Medical history)', 'label': 'Đã diệt H. Pylori (H. pylori Eradicated)', 'type': 'bool'},
    'Med_H_pylori_Status_Negative': {'group': 'Dịch tễ và Tiền sử (Epidemiology and Medical history)', 'label': 'H. Pylori Âm tính (H. pylori Negative)', 'type': 'bool'},

    'Tongue_Color_Red': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Chất lưỡi đỏ (Red tongue)', 'type': 'bool'},
    'Tongue_Color_Pale_red': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Chất lưỡi hồng (Pale red tongue)', 'type': 'bool'},
    'Tongue_Color_Pale': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Chất lưỡi nhợt (Pale tongue)', 'type': 'bool'},
    'Tongue_Color_Bluish_purple': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Chất lưỡi tím/xanh (Bluish purple tongue)', 'type': 'bool'},
    'Tongue_Shape_Normal': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Hình dáng lưỡi bình thường (Normal shape)', 'type': 'bool'},
    'Tongue_Shape_Thin': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Lưỡi thon/gầy (Thin tongue)', 'type': 'bool'},
    'Tongue_Moisture_Wet': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Lưỡi ướt (Wet tongue)', 'type': 'bool'},
    'Tongue_Moisture_Moist': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Lưỡi nhuận (Moist tongue)', 'type': 'bool'},
    'Tongue_Moisture_Dry': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Lưỡi khô (Dry tongue)', 'type': 'bool'},
    'Tongue_Cracked': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Đường nứt lưỡi (Cracked tongue)', 'type': 'bool'},
    'Tongue_Scalloped': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Dấu ấn răng (Scalloped tongue)', 'type': 'bool'},
    'Tongue_Stasis_Spots': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Điểm ứ huyết (Stasis spots)', 'type': 'bool'},
    'Tongue_Sublingual_Veins': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Giãn tĩnh mạch dưới lưỡi (Sublingual veins)', 'type': 'bool'},
    'Tongue_Movement_Flexible': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Lưỡi cử động linh hoạt (Flexible movement)', 'type': 'bool'},
    'Coating_Color_White': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Rêu trắng (White coating)', 'type': 'bool'},
    'Coating_Color_White_yellow': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Rêu trắng vàng (White-yellow coating)', 'type': 'bool'},
    'Coating_Thickness_Thick': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Rêu dày (Thick coating)', 'type': 'bool'},
    'Coating_Thickness_Thin': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Rêu mỏng (Thin coating)', 'type': 'bool'},
    'Coating_Greasy': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Rêu nhờn (Greasy coating)', 'type': 'bool'},
    'Coating_Rotten': {'group': 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'label': 'Rêu mục nát (Rotten coating)', 'type': 'bool'},

    'Pulse_Wiry': {'group': 'Khám Mạch (Pulse diagnosis)', 'label': 'Mạch Huyền (Wiry pulse)', 'type': 'bool'},
    'Pulse_Slippery': {'group': 'Khám Mạch (Pulse diagnosis)', 'label': 'Mạch Hoạt (Slippery pulse)', 'type': 'bool'},
    'Pulse_Rapid': {'group': 'Khám Mạch (Pulse diagnosis)', 'label': 'Mạch Sác (Rapid pulse)', 'type': 'bool'},
    'Pulse_Thready': {'group': 'Khám Mạch (Pulse diagnosis)', 'label': 'Mạch Tế (Thready pulse)', 'type': 'bool'},
    'Pulse_Choppy': {'group': 'Khám Mạch (Pulse diagnosis)', 'label': 'Mạch Sáp (Choppy pulse)', 'type': 'bool'},

    'Sym_Epigastric_Pain': {'group': 'Triệu chứng (Symptom)', 'label': 'Đau thượng vị (Epigastric pain)', 'type': 'bool'},
    'Sym_Fixed_Epigastric_Pain': {'group': 'Triệu chứng (Symptom)', 'label': 'Đau cố định (Fixed epigastric pain)', 'type': 'bool'},
    'Sym_Dull_Epigastric_Pain': {'group': 'Triệu chứng (Symptom)', 'label': 'Đau âm ỉ thượng vị (Dull epigastric pain)', 'type': 'bool'},
    'Sym_Hypochondriac_Pain': {'group': 'Triệu chứng (Symptom)', 'label': 'Đau hạ sườn (Hypochondriac pain)', 'type': 'bool'},
    'Sym_Epigastric_Discomfort': {'group': 'Triệu chứng (Symptom)', 'label': 'Khó chịu thượng vị (Epigastric discomfort)', 'type': 'bool'},
    'Sym_Epigastric_Burning': {'group': 'Triệu chứng (Symptom)', 'label': 'Nóng rát thượng vị (Epigastric burning)', 'type': 'bool'},
    'Sym_Postprandial_Fullness_Pain': {'group': 'Triệu chứng (Symptom)', 'label': 'Đầy tức, đau sau ăn (Postprandial fullness/pain)', 'type': 'bool'},
    'Sym_Pain_Relieved_By_Pressure': {'group': 'Triệu chứng (Symptom)', 'label': 'Đau thiện án / Giảm khi ấn (Pain relieved by pressure)', 'type': 'bool'},
    'Sym_Pain_Aggravated_By_Pressure': {'group': 'Triệu chứng (Symptom)', 'label': 'Đau cự án / Tăng khi ấn (Pain aggravated by pressure)', 'type': 'bool'},
    'Sym_Preference_For_Warmth': {'group': 'Triệu chứng (Symptom)', 'label': 'Thích ấm (Preference for warmth)', 'type': 'bool'},
    'Sym_Triggered_By_Emotion': {'group': 'Triệu chứng (Symptom)', 'label': 'Khởi phát do cảm xúc (Triggered by emotion)', 'type': 'bool'},
    'Sym_Belching': {'group': 'Triệu chứng (Symptom)', 'label': 'Ợ hơi (Belching)', 'type': 'bool'},
    'Sym_Acid_Regurgitation': {'group': 'Triệu chứng (Symptom)', 'label': 'Ợ chua (Acid regurgitation)', 'type': 'bool'},
    'Sym_Nausea': {'group': 'Triệu chứng (Symptom)', 'label': 'Buồn nôn (Nausea)', 'type': 'bool'},
    'Sym_Vomiting_Clear_Fluid': {'group': 'Triệu chứng (Symptom)', 'label': 'Nôn ra nước trong (Vomiting clear fluid)', 'type': 'bool'},
    'Sym_Poor_Appetite': {'group': 'Triệu chứng (Symptom)', 'label': 'Ăn kém (Poor appetite)', 'type': 'bool'},
    'Sym_Hunger_Without_Appetite': {'group': 'Triệu chứng (Symptom)', 'label': 'Đói nhưng không muốn ăn (Hunger without appetite)', 'type': 'bool'},
    'Sym_Dry_Mouth': {'group': 'Triệu chứng (Symptom)', 'label': 'Khô miệng (Dry mouth)', 'type': 'bool'},
    'Sym_Bitter_Taste': {'group': 'Triệu chứng (Symptom)', 'label': 'Miệng đắng (Bitter taste)', 'type': 'bool'},
    'Sym_Halitosis': {'group': 'Triệu chứng (Symptom)', 'label': 'Hôi miệng (Halitosis)', 'type': 'bool'},
    'Sym_Loose_Stools': {'group': 'Triệu chứng (Symptom)', 'label': 'Phân lỏng (Loose stools)', 'type': 'bool'},
    'Sym_Constipation': {'group': 'Triệu chứng (Symptom)', 'label': 'Táo bón (Constipation)', 'type': 'bool'},
    'Sym_Melena': {'group': 'Triệu chứng (Symptom)', 'label': 'Phân đen (Melena)', 'type': 'bool'},
    'Sym_Fatigue': {'group': 'Triệu chứng (Symptom)', 'label': 'Mệt mỏi (Fatigue)', 'type': 'bool'},
    'Sym_Short_Breath_Lazy_Speech': {'group': 'Triệu chứng (Symptom)', 'label': 'Đoản khí, lười nói (Short breath & lazy speech)', 'type': 'bool'},
    'Sym_Chest_Tightness': {'group': 'Triệu chứng (Symptom)', 'label': 'Tức ngực (Chest tightness)', 'type': 'bool'},
    'Sym_Irritability': {'group': 'Triệu chứng (Symptom)', 'label': 'Dễ cáu gắt (Irritability)', 'type': 'bool'},
    'Sym_Dark_Complexion': {'group': 'Triệu chứng (Symptom)', 'label': 'Sắc mặt sạm tối (Dark complexion)', 'type': 'bool'}
}

# ==========================================
# 3. TẢI MÔ HÌNH, SCALER VÀ EXPLAINER
# ==========================================
@st.cache_resource
def load_assets():
    scaler_1 = joblib.load('CDSS_Feature_Scaler.pkl')
    model_1  = joblib.load('Final_CDSS_Screening_Model.pkl')

    scaler_2 = joblib.load('Phase2_CDSS_Feature_Scaler.pkl')
    model_2  = joblib.load('Phase2_Final_CDSS_MultiLabel_Model.pkl')

    # Background cho KernelExplainer (tạo bởi make_background.py)
    try:
        bg_1 = joblib.load('shap_background_phase1.pkl')
    except FileNotFoundError:
        st.error("❌ Thiếu file 'shap_background_phase1.pkl'. Hãy chạy script 'make_background.py' một lần để tạo.")
        st.stop()

    # ✅ KernelExplainer chạy trên CHÍNH model_1, đầu ra = P(class 1 = AG)
    #    → giải thích luôn nhất quán với kết quả sàng lọc, không phụ thuộc loại model
    explainer_1 = shap.KernelExplainer(
        lambda X: model_1.predict_proba(X)[:, 1],
        bg_1['scaled'].values
    )

    return scaler_1, model_1, scaler_2, model_2, explainer_1

scaler_1, model_1, scaler_2, model_2, explainer_1 = load_assets()

try:
    with open('List_of_Final_K_Features.txt', 'r') as f:
        feat_1_raw = [line.strip() for line in f.readlines()]
    with open('Phase2_List_of_Final_K_Features.txt', 'r') as f:
        feat_2_raw = [line.strip() for line in f.readlines()]
    # ✅ Thứ tự cột xác suất của model_2 = thứ tự trong file này (sinh ra từ mlb.classes_)
    with open('Phase2_Target_Columns.txt', 'r') as f:
        raw_order = [line.strip().replace('Target_TCM_', '') for line in f]
except FileNotFoundError as e:
    st.error(f"❌ Lỗi hệ thống: Không tìm thấy file danh sách. Chi tiết: {e}")
    st.stop()

# ✅ Ánh xạ tên thô → tên hiển thị, giữ NGUYÊN thứ tự của mô hình
SYNDROME_DISPLAY = {
    'Liver_Stomach_Depressed_Heat': 'Can Vị Uất Nhiệt (Liver Stomach Depressed Heat)',
    'Liver_Stomach_Qi_Stagnation': 'Can Vị Khí Trệ (Liver Stomach Qi Stagnation)',
    'Spleen_Stomach_Damp_Heat': 'Tỳ Vị Thấp Nhiệt (Spleen Stomach Damp Heat)',
    'Spleen_Stomach_Deficiency': 'Tỳ Vị Hư Hàn (Spleen Stomach Deficiency)',
    'Stomach_Collateral_Stasis': 'Vị Lạc Ứ Huyết (Stomach Collateral Stasis)',
    'Stomach_Yin_Deficiency': 'Vị Âm Hư (Stomach Yin Deficiency)',
}
syndrome_names = [SYNDROME_DISPLAY.get(c, c) for c in raw_order]

# Kiểm chứng MỘT lần (xong có thể xóa): 2 con số phải bằng nhau (= 6)
# st.caption(f"*Debug: {len(syndrome_names)} chứng hậu | {len(model_2.estimators_)} estimators*")

all_unique_features = sorted(list(set(feat_1_raw + feat_2_raw)))

# ==========================================
# 4. GIAO DIỆN NHẬP LIỆU (SIDEBAR)
# ==========================================
st.sidebar.header("📋 Nhập Thông Tin Lâm Sàng (Enter clinical information)")
user_inputs = {}

with st.sidebar.form(key='patient_form'):
    groups = ['Dịch tễ và Tiền sử (Epidemiology and Medical history)', 'Lưỡi và Rêu lưỡi (Tongue and Coating)', 'Khám Mạch (Pulse diagnosis)', 'Triệu chứng (Symptom)']

    for grp in groups:
        st.markdown(f"**{grp}**")
        for feat in all_unique_features:
            if feat in FEATURE_DICT and FEATURE_DICT[feat]['group'] == grp:
                label = FEATURE_DICT[feat]['label']
                if FEATURE_DICT[feat]['type'] == 'number':
                    user_inputs[feat] = st.number_input(label, min_value=18, max_value=100, value=45, step=1)
                else:
                    user_inputs[feat] = st.checkbox(label, value=False)
        st.markdown("---")

    submit_button = st.form_submit_button(label='🚀 Phân tích Dữ liệu (Clinical data analysis)')

# ==========================================
# 5. LUỒNG XỬ LÝ CHẨN ĐOÁN
# ==========================================
if submit_button:
    input_numeric = {k: (int(v) if isinstance(v, bool) else v) for k, v in user_inputs.items()}

    with st.spinner('Hệ thống đang xử lý dữ liệu (System processing data)...'):
        # -----------------------------------
        # PHASE 1: SÀNG LỌC VIÊM TEO (AG)
        # -----------------------------------
        scaler_1_feats = list(scaler_1.feature_names_in_) if hasattr(scaler_1, 'feature_names_in_') else feat_1_raw
        model_1_feats  = list(model_1.feature_names_in_)  if hasattr(model_1, 'feature_names_in_')  else feat_1_raw

        # Ép cột theo Scaler -> Scale -> Ép cột theo Model
        data_for_scaler_1 = pd.DataFrame([input_numeric]).reindex(columns=scaler_1_feats, fill_value=0)
        scaled_array_1 = scaler_1.transform(data_for_scaler_1)
        scaled_df_1 = pd.DataFrame(scaled_array_1, columns=scaler_1_feats)

        X_p1_final = scaled_df_1.reindex(columns=model_1_feats)
        data_p1_raw_final = data_for_scaler_1.reindex(columns=model_1_feats)

        # Dự đoán (column 1 = class dương theo nhãn train)
        prob_ag = model_1.predict_proba(X_p1_final)[0][1]
        threshold_ag = 0.4081
        is_ag_positive = prob_ag >= threshold_ag

        st.subheader("🔴 1. Kết quả Sàng lọc Viêm teo niêm mạc dạ dày")
        st.markdown("*Atrophic Gastritis Screening Results*")
        if is_ag_positive:
            st.error(
                f"**CẢNH BÁO CAO:** Bệnh nhân có nguy cơ mắc Viêm teo dạ dày (Xác suất: {prob_ag * 100:.1f}%)\n\n"
                f"*HIGH WARNING: Patient is at risk for Atrophic Gastritis (Probability: {prob_ag * 100:.1f}%)*"
            )
        else:
            st.success(
                f"**NGUY CƠ THẤP:** Xác suất mắc Viêm teo dạ dày là {prob_ag * 100:.1f}%\n\n"
                f"**LOW RISK:** The probability of Atrophic Gastritis is {prob_ag * 100:.1f}%*"
            )

        # ==========================================
        # MODULE XAI: AI TỰ ĐỘNG GIẢI THÍCH (SHAP)
        # ==========================================
        with st.expander(
                "🔍 Nhấn vào đây để xem AI giải thích cơ chế chẩn đoán cho bệnh nhân này\n\n"
                "*Click here to view the AI explanation of the diagnostic mechanism for this patient*",
                expanded=True
        ):

            # SHAP tính trên CHÍNH model_1, cùng không gian dữ liệu (scaled)
            shap_values = np.asarray(
                explainer_1.shap_values(X_p1_final.values, nsamples=500, silent=True)
            ).reshape(-1)
            expected_val = float(explainer_1.expected_value)

            # Debug: 2 con số này phải gần như trùng nhau (cùng 1 model, 1 đầu ra)
            # st.caption(f"*Debug nội bộ: prob_ag={prob_ag:.3f} | base + ΣSHAP={expected_val + shap_values.sum():.3f}*")

            st.markdown("##### 📝 Giải thích bằng ngôn ngữ y khoa (Explanation in medical terms):")
            symptoms_pushing_up = []
            symptoms_pushing_down = []
            symptoms_pushing_up_en = []  # ← danh sách tiếng Anh
            symptoms_pushing_down_en = []  # ← danh sách tiếng Anh

            for i, feat_name in enumerate(model_1_feats):
                full_label = FEATURE_DICT.get(feat_name, {}).get('label', feat_name)
                friendly_name = full_label.split(' (')[0]  # Tên tiếng Việt
                eng_name = full_label.split(' (')[-1].rstrip(
                    ')') if ' (' in full_label else friendly_name  # Tên tiếng Anh
                val_raw = data_p1_raw_final.iloc[0, i]
                shap_val = shap_values[i]

                if val_raw > 0:
                    if shap_val > 0.01:
                        if FEATURE_DICT.get(feat_name, {}).get('type') == 'number':
                            symptoms_pushing_up.append(f"Độ tuổi ({val_raw})")
                            symptoms_pushing_up_en.append(f"Age ({val_raw})")
                        else:
                            symptoms_pushing_up.append(friendly_name)
                            symptoms_pushing_up_en.append(eng_name)
                    elif shap_val < -0.01:
                        if FEATURE_DICT.get(feat_name, {}).get('type') == 'number':
                            symptoms_pushing_down.append(f"Độ tuổi ({val_raw})")
                            symptoms_pushing_down_en.append(f"Age ({val_raw})")
                        else:
                            symptoms_pushing_down.append(friendly_name)
                            symptoms_pushing_down_en.append(eng_name)

            if symptoms_pushing_up:
                st.warning(
                    f"**Các đặc điểm làm TĂNG nguy cơ Viêm teo:** {', '.join(symptoms_pushing_up)}.\n\n"
                    f"*Features that INCREASE the risk of Atrophic Gastritis: {', '.join(symptoms_pushing_up_en)}.*"
                )
            if symptoms_pushing_down:
                st.success(
                    f"**Các đặc điểm giúp KÌM HÃM/GIẢM nguy cơ Viêm teo:** {', '.join(symptoms_pushing_down)}.\n\n"
                    f"*Features that INHIBIT/REDUCE the risk of Atrophic Gastritis: {', '.join(symptoms_pushing_down_en)}.*"
                )
            if not symptoms_pushing_up and not symptoms_pushing_down:
                st.info(
                    "Bệnh nhân không có dấu hiệu lâm sàng nào nổi bật tác động đến mô hình.\n\n"
                    "*The patient presents with no prominent clinical signs impacting the model.*"
                )

            st.markdown("##### 📊 Biểu đồ trọng số (SHAP Waterfall):")


            # ✅ Hàm tạo nhãn song ngữ Việt - Anh
            def bilingual_label(feat):
                label = FEATURE_DICT.get(feat, {}).get('label', feat)
                if ' (' in label:
                    vi, en = label.split(' (', 1)
                    return f"{vi} | {en.rstrip(')')}"
                return label


            display_feature_names = [bilingual_label(f) for f in model_1_feats]

            patient_explanation = shap.Explanation(
                values=shap_values,
                base_values=expected_val,
                data=data_p1_raw_final.iloc[0].values,  # vẫn hiển thị 0/1 gốc
                feature_names=display_feature_names
            )

            fig, ax = plt.subplots(figsize=(11, 5))  # nới rộng khung để nhãn song ngữ không bị chật
            shap.plots.waterfall(patient_explanation, show=False, max_display=8)
            plt.yticks(fontsize=9)  # thu nhỏ chữ trục Y một chút cho thoáng
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")

        # -----------------------------------
        # PHASE 2: CHẨN ĐOÁN CHỨNG HẬU
        # -----------------------------------
        st.subheader("☯️ 2. Biện chứng Y học cổ truyền (Mô hình Đa nhãn)")
        st.markdown("*Traditional Medicine Syndrome Differentiation (Multi-label Model)*")
        if is_ag_positive:
            scaler_2_feats = list(scaler_2.feature_names_in_) if hasattr(scaler_2, 'feature_names_in_') else feat_2_raw
            model_2_feats  = list(model_2.feature_names_in_)  if hasattr(model_2, 'feature_names_in_')  else feat_2_raw

            data_for_scaler_2 = pd.DataFrame([input_numeric]).reindex(columns=scaler_2_feats, fill_value=0)
            scaled_array_2 = scaler_2.transform(data_for_scaler_2)
            scaled_df_2 = pd.DataFrame(scaled_array_2, columns=scaler_2_feats)
            X_p2_final = scaled_df_2.reindex(columns=model_2_feats)

            probs_tcm = model_2.predict_proba(X_p2_final)[0]

            cols = st.columns(3)
            for i, (syndrome, prob) in enumerate(zip(syndrome_names, probs_tcm)):
                col = cols[i % 3]
                with col:
                    if prob >= 0.5:
                        st.warning(f"**{syndrome}**")
                        st.progress(float(prob), text=f"Khả năng mắc (Probability): {prob * 100:.1f}%")
                    else:
                        st.info(f"*{syndrome}*")
                        st.progress(float(prob), text=f"Khả năng mắc (Probability): {prob * 100:.1f}%")
        else:
            st.info(
                    "Vì bệnh nhân có nguy cơ Viêm teo thấp, hệ thống ưu tiên theo dõi sức khỏe định kỳ và chưa kích hoạt chẩn đoán hội chứng Y học cổ truyền chuyên sâu.\n\n"
                    "*Due to the patient's low risk of Atrophic Gastritis, the system prioritizes routine health monitoring and has not activated advanced Traditional Medicine syndrome differentiation.*"
                    )