# Nova Canvas 이미지 품질 개선 솔루션 — 기술 소개 및 사용 가이드



##  기술: PromptSculptor + Maestro 결합

  

### 2-1. PromptSculptor 

- **PromptSculptor**: "Multi-Agent Based Text-to-Image Prompt Optimization" (EMNLP 2025 System Demonstration Track) — https://arxiv.org/abs/2509.12446

  

PromptSculptor는 짧고 모호한 사용자 프롬프트를 **전문가 에이전트들이 협업하여 고품질 프롬프트로 변환**하는 멀티에이전트 프레임워크입니다.

  

본 솔루션에서는 이를 다음과 같이 구현했습니다.

  

**Nova Pro (스타일 분석 전문가)**

- 레퍼런스 이미지를 직접 분석하여 색상 팔레트(hex 코드), 아트 스타일, 구도를 추출

- 유저가 직접 선택한 색상이 있으면 해당 색상을 "MANDATORY" 팔레트로 우선 적용

- 레퍼런스 이미지는 색상이 아닌 스타일/구도 참고용으로만 사용

- 할로윈 pumpkin = 오렌지가 아닌 팔레트 색상, ghost = 흰색이 아닌 팔레트 색상 등 기본 색상 override

- "NO black outlines, NO dark borders, elements filled with palette colors only" 등 렌더링 규칙 추가

  

**Nova Lite (핵심 요소 추출 전문가)**

- 사용자 프롬프트에서 가장 중요한 시각 요소 5~8개만 선별

- 유저 선택 색상을 "MANDATORY" 팔레트로 받아 각 요소에 명시적으로 적용

- 할로윈, 크리스마스 등 테마에 관계없이 지정된 팔레트 색상만 사용

- 요소 개수와 크기를 이미지에 비례하도록 조정

  

**Nova Pro (결합 전문가)**

- 위 두 프롬프트를 하나의 최종 프롬프트로 통합

- 색상 제약 + 요소 배치 + 렌더링 규칙을 모두 포함

- 950자 이내로 최적화

  

```

사용자 프롬프트 + 레퍼런스 이미지

↓

[Nova Pro] 스타일 분석 → 색상 강제 프롬프트

[Nova Lite] 핵심 요소 추출 → 간결한 프롬프트

↓

[Nova Pro] 두 프롬프트 결합 → 최종 프롬프트

↓

Nova Canvas 이미지 생성

```

  

### 2-2. Maestro 
- **Maestro**: "Self-Improving Text-to-Image Generation via Agent Orchestration" — https://arxiv.org/abs/2509.10704
  

Maestro는 생성된 이미지를 자율적으로 평가하고 프롬프트를 반복 개선하는 **자기 진화 시스템**입니다.

  

본 솔루션에서 구현한 Maestro의 핵심 요소는 다음과 같습니다.

  

**DVQ (Decomposed Visual Questions)**

- 사용자 프롬프트와 레퍼런스 이미지를 분석하여 8~10개의 Yes/No 평가 질문 자동 생성

- 예시: "Is there a Christmas tree visible in the image?", "Is the color palette predominantly dark teal?"

- 레퍼런스 이미지가 있으면 해당 이미지의 실제 색상을 기준으로 질문 생성 (고정 색상 기준 없음)

- 한 번 생성 후 전체 파이프라인에서 재사용

  

**DVQ 평가 (VQA)**

- Nova Pro Vision이 생성된 이미지를 보고 각 DVQ에 Yes/No로 답변

- 색상·스타일 관련 질문은 레퍼런스 이미지와 나란히 비교하여 평가

- DVQ 점수 = Yes 항목 수 / 전체 DVQ 수

  

**Pairwise 비교 (Position Bias 제거)**

- 현재 생성 이미지 vs 이전 최고 이미지를 Nova Pro가 직접 비교

- Position bias 방지를 위해 순서를 바꿔 2n번(기본 6회) 비교 후 다수결

- 레퍼런스 이미지가 있으면 "레퍼런스와 더 유사한 이미지"를 주 기준으로 판단

  

**Best 선택 로직**

- DVQ 점수가 5%p 이상 높으면 Pairwise 없이 즉시 업데이트

- DVQ 점수가 비슷하면 Pairwise 비교로 결정

- DVQ 점수가 5%p 이상 낮으면 기존 best 유지 (품질 하락 방지)

  

---

  

## 3. 전체 파이프라인

  

```

[STEP 1] 색상 추출

- 유저가 직접 선택한 색상 우선 사용

- 없으면 레퍼런스 이미지에서 k-means로 자동 추출 (최대 10개 hex)

  

[STEP 2] PromptSculptor (매 iteration 실행)

- Nova Pro + Nova Lite가 각자 프롬프트 재창조

- Nova Pro가 두 프롬프트를 결합하여 최종 프롬프트 생성

  

[STEP 3] DVQ 생성 (1회, 전체 재사용)

- 최종 프롬프트 + 레퍼런스 이미지 기반으로 평가 질문 생성

  

[STEP 4] Nova Canvas 이미지 생성

- COLOR_GUIDED_GENERATION: 색상 팔레트 강제 적용

- 유저 색상 선택 시: colors만 사용 (referenceImage 제외)

- 자동 추출 시: colors + referenceImage 동시 사용

  

[STEP 5] DVQ 평가

- Nova Pro Vision이 각 질문에 Yes/No 답변

- 스타일/색상 질문은 레퍼런스와 비교 평가

  

[STEP 6] Best 선택

- DVQ 점수 차이 기반 + Pairwise 비교 결합

  

[STEP 7] 종료 판단

- DVQ 점수 90% 이상: 조기 종료

- patience 횟수 연속 미개선: 종료

- max iteration 도달: 종료

  

[STEP 8] 다음 iteration → STEP 2로 반복

```

  

---

  

## 4. AWS 기술 스택

  

| 역할 | 모델/서비스 |
|------|------------|
| 이미지 생성 | Amazon Nova Canvas (`amazon.nova-canvas-v1:0`) |
| 프롬프트 최적화 (스타일 분석) | Amazon Nova Pro (`us.amazon.nova-pro-v1:0`) |
| 프롬프트 최적화 (요소 추출) | Amazon Nova Lite (`us.amazon.nova-lite-v1:0`) |
| DVQ 생성 / 평가 / Pairwise 비교 | Amazon Nova Pro Vision |


  

---

  

## 5. UI 사용 가이드

  

### 화면 구성

  

좌측 사이드바에서 모드와 설정을 선택하고, 우측 메인 화면에서 결과를 확인합니다.

  

### 모드 선택

  

| 모드 | 설명 |

|------|------|

| Generation | 텍스트 프롬프트만으로 이미지 생성 (Nova Canvas / Titan v2) |

| Variation | 레퍼런스 이미지 + 프롬프트로 변형 이미지 생성 |

| Optimize (Nova) | PromptSculptor + Maestro 파이프라인으로 자동 최적화 |

  

### Optimize 모드 설정 항목

  

**레퍼런스 이미지 업로드**

- 생성하고 싶은 스타일/색상의 기준 이미지를 업로드합니다.

- 업로드하면 색상 팔레트가 자동으로 추출되어 컬러 피커에 반영됩니다.

  

**컬러 팔레트 (5개 색상 피커)**

- 레퍼런스에서 자동 추출된 색상이 기본값으로 설정됩니다.

- 원하는 색상으로 직접 수정할 수 있습니다.

- 선택한 색상은 Nova Pro/Lite에 "MANDATORY 팔레트"로 전달되어 모든 요소(배경, 아이콘, 선, 노드)에 강제 적용됩니다.

- 할로윈 pumpkin이라도 오렌지가 아닌 선택한 색상으로 렌더링됩니다.

- 선택한 색상이 Nova Canvas의 `COLOR_GUIDED_GENERATION`에도 직접 전달됩니다.

  

**패턴 밀도**

- `very sparse`: 선 3~5개, 넓은 여백 (레퍼런스 원본에 가장 가까운 설정)

- `sparse`: 선 몇 개, 원형 노드, 여백 있음 (권장)

- `moderate`: 중간 밀도

- `dense`: 전체를 채우는 촘촘한 패턴

  

**요소 개수 (3~12)**

- 이미지에 들어갈 아이콘/이모지의 최대 개수를 설정합니다.

- 적을수록 각 요소가 크고 명확하게 표현됩니다.

- 권장값: 5~7개

  

**Max Iterations**

- 프롬프트 재창조 및 이미지 생성 반복 횟수입니다.

- 높을수록 더 많은 시도를 하지만 시간과 비용이 증가합니다.

- 권장값: 3~5회

  

**Patience**

- 연속으로 개선이 없을 때 자동 종료하는 횟수입니다.

- 권장값: 2~3회

  

**Width / Height**

- 생성 이미지 해상도입니다. 기본값 1024×1024.

  

### 결과 화면

  

**좌측 패널**

- Original Prompt: 입력한 원본 프롬프트

- PromptSculptor Output: Nova Pro / Nova Lite / 결합 프롬프트 각각 표시

- Final Prompt: 최종 생성에 사용된 프롬프트

- DVQs: 자동 생성된 평가 질문 목록

- Final DVQ Evaluation: 최종 이미지의 각 질문별 ✅/❌ 결과

- DVQ Score: 전체 점수 (%)

  

**우측 패널**

- Best Result: 전체 iteration 중 가장 높은 점수의 이미지

- Download Best Image: 최고 결과 이미지 다운로드

- Iteration History: 각 iteration별 이미지, DVQ 점수, 사용된 프롬프트 확인 가능

  

### 사용 팁

  

1. 레퍼런스 이미지를 업로드하면 색상이 자동 추출되므로, 컬러 피커를 별도로 수정하지 않아도 됩니다.

2. 프롬프트는 짧고 추상적으로 입력해도 됩니다. (예: "Halloween atmosphere", "Christmas mood") Nova Pro가 자동으로 구체적인 요소로 확장합니다.

3. 패턴이 너무 복잡하게 나오면 패턴 밀도를 `very sparse`로 낮추세요.

4. 이모지/아이콘이 너무 많으면 요소 개수를 4~5로 줄이세요.

5. Iteration History에서 각 시도의 결과를 비교하고, 마음에 드는 이미지를 개별 다운로드할 수 있습니다.

  

---

  
