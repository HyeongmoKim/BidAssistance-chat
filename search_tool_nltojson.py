from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Literal
import json
from langchain_core.messages import SystemMessage, HumanMessage

OpFrom = Literal["gte", "gt", "eq"]
OpTo   = Literal["lte", "lt"]

AggOp  = Literal["min", "max", "avg", "sum", "count"]
OutType = Literal["all", "field", "agg", "error"]

DateBase = Literal["startDate", "endDate", "openDate"]
Kind = Literal["absolute", "calendar"]
Unit = Literal["day", "week", "month", "year"]
Position = Literal["start", "end"]

class RangeFrom(BaseModel):
    op: OpFrom
    value: float


class RangeTo(BaseModel):
    op: OpTo
    value: float


class Range(BaseModel):
    from_: Optional[RangeFrom] = Field(None, alias="from")
    to: Optional[RangeTo] = None

    @model_validator(mode="after")
    def check_range(cls, values:dict)->dict:
        f, t = values.from_, values.to

        if not f and not t:
            return values

        # eq는 단독 비교만 허용
        if f and f.op == "eq" and t is not None:
            raise ValueError("eq comparison must not include to")

        return values

class TimePoint(BaseModel):
    kind: Kind
    value: Optional[int]
    unit: Optional[Unit]
    offset: Optional[int]
    position: Optional[Position]

    @model_validator(mode="after")
    def check_kind_rules(self):

        if self.kind == "absolute":
            if self.value is None:
                raise ValueError("absolute kind requires value")
            if any(
                getattr(self, field) is not None
                for field in ("unit", "offset", "position")
            ):
                raise ValueError("absolute kind cannot have unit/offset/position")

        if self.kind == "calendar":
            if self.value is not None:
                raise ValueError("calendar kind cannot have value")
            if (
                self.unit is None
                and self.offset is None
                and self.position is None
            ):
                raise ValueError(
                    "calendar kind requires at least one of unit/offset/position"
                )

        return self

class TimeRange(BaseModel):
    base: DateBase
    from_: Optional[TimePoint] = Field(None, alias="from")
    to: Optional[TimePoint] = None


class Filter(BaseModel):
    bidRealId: Optional[str]
    name: Optional[str]
    region: Optional[str]
    organization: Optional[str]

    estimatePrice: Optional[Range]
    basicPrice: Optional[Range]
    minimumBidRate: Optional[Range]
    bidRange: Optional[Range]

    timeRange: Optional[TimeRange]

    
class OutputItem(BaseModel):
    type: OutType
    field: Optional[str]
    op: Optional[AggOp]

    @model_validator(mode="after")
    def validate_output(cls, values:dict)->dict:
        t, field, op = values.type, values.field, values.op

        if t == "all":
            if field is not None or op is not None:
                raise ValueError("all type must have null field and op")

        if t == "field":
            if field is None or op is not None:
                raise ValueError("field type requires field and null op")

        if t == "agg":
            if op is None:
                raise ValueError("agg type requires op")
            if op != "count" and field is None:
                raise ValueError("agg type requires field unless op is count")

        if t == "error":
            if field is not None or op is not None:
                raise ValueError("error type must have null field and op")

        return values
    
class Query(BaseModel):
    limit: Optional[int]=None
    filter: Optional[Filter]=None
    output: List[OutputItem]

    @model_validator(mode="after")
    def final_checks(cls, values):
        output = values.output
        limit = values.limit
        filter_ = values.filter

        #output은 비면 안됨
        if not output:
            raise ValueError("output cannot be empty")
        
        #types = {o.type for o in output}
        
        has_agg = any(o.type == "agg" for o in output)
        has_non_agg = any(o.type in ("all", "field") for o in output)

        # agg + all/field 혼합 금지
        if has_agg and has_non_agg:
          raise ValueError("agg cannot be mixed with all/field")

        # error면 limit은 null
        if any(o.type == "error" for o in output):
          if limit is not None:
              raise ValueError("limit must be null when output type is error")
          return values
        
        if filter_ is None:
            raise ValueError("filter is required for notice query")

        # agg only → limit null
        if has_agg:
          if limit is not None:
              raise ValueError("limit must be null for agg queries")
        else:
            # 일반 조회는 limit 필수
            if limit is None:
                raise ValueError("limit is required for non-agg queries")

        return values
    
llm = ChatOpenAI(model="gpt-5-nano", temperature=1)

@tool
def extract_notice_query(user_query: str) -> dict:
    """
    이 도구는 사용자의 질문을 공고 조회를 위한 필터 조건 JSON 객체로 변환한다.
    공고 조회가 아닌 질문에는 사용하지 않는다.
    """
    prompt="""
    사용자의 질문을 공고 조회를 위한 필터 조건 json 객체로 변환한다.
    반드시 아래 규칙에 따라 JSON 객체 하나만 생성해야 한다.

    중요 규칙:
    - 출력 JSON의 필드명은 반드시 아래에 정의된 이름만 사용해야 한다.
    - 정의되지 않은 필드명을 새로 만들면 안 된다.
    - filter에서 명시하지 않는 필드는 반드시 null로 설정한다.
    - output은 항상 최소 1개 이상의 객체를 포함해야 한다.
    - 출력은 반드시 JSON 객체 하나만 반환한다. 설명 문장, 주석, 자연어는 절대 포함하지 않는다.
    - 시간 관련 조건은 반드시 timeRange로만 표현한다.
    - 필드의 의미를 추론해 임의로 확장하거나 축약하지 않는다.
    - 숫자형 날짜는 yyyyMMddHHmm 형식만 사용한다. (예: 202601231359)
    - LLM은 날짜 계산을 하지 않고 반드시 선언형 구조로만 표현한다.

    --------------------------------------------------
    name 판단 규칙:
    - 스키마에 존재하지 않는 일반 텍스트 조건은 name 필드에 포함 검색으로 처리한다.
    - 단순 키워드 성격의 일반 텍스트 조건은 name 필드에 포함 검색으로 처리한다.
    - 숫자, 단위(년, 개월, 억, %, 등), 비교 표현(이상, 이하, 초과, 미만) 중 하나라도 포함된 조건은 단순 키워드로 간주하지 않는다.
    - 위 조건이 스키마 필드에 직접 매핑되지 않는 경우, 반드시 error로 처리한다.
    - 해당 조건을 name 필드로 포함 검색 처리하지 않는다.
    - 예: 여성기업 대상 공고 -> name = "여성기업" , 창업지원금 1억 이상 -> error 처리

    timeRange (시간 조건 선언)
    
    timeRange는 “언제의 공고를 조회할 것인가”를
    계산 없이 선언적으로 표현하는 필드이다.

    timeRange는 하나만 설정할 수 있다.

    기준 필드(base):
    - startDate : 공고 시작일 기준 조회
    - endDate   : 공고 마감일 기준 조회
    - openDate  : 공고 개찰일 기준 조회

    base 판단 규칙:

    1. 사용자가 명시적으로 언급한 날짜 기준이 있으면 그것을 최우선으로 사용한다.
      - "시작", "시작일", "등록" → startDate
      - "마감", "마감일", "마감 기준" → endDate
      - "개찰", "개찰일" → openDate

    2. "까지", "부터", "이전", "이후", "이내" 같은 표현은
      기간의 방향·범위를 나타낼 뿐 base를 결정하지 않는다.

    3. 명시적인 기준이 없는 경우 (예: “언제부터 언제까지 공고 보여줘”) base는 endDate로 설정한다.
       output에 endDate/startDate/openDate가 포함되어 있어도 timeRange.base를 결정하는 근거로 사용하지 않는다. base는 오직 사용자가 “마감일 기준”, “개찰일 기준”처럼 날짜 기준을 직접 언급했을 때만 설정한다.

    4. 둘 이상의 기준이 동시에 언급된 경우, 문장에서 직접 수식되는 기준을 우선한다.

    - “~부터” → from 사용
    - “~까지” → to 사용
    - from과 to는 각각 독립적으로 설정한다.

    - from 또는 to 중 하나만 존재해도 된다.

    - from과 to는 서로 다른 kind를 가질 수 있다.

    - 단, from 내부 / to 내부에서는 kind를 혼용할 수 없다.



    -from 또는 to는 숫자 날짜 또는 calendar 선언 객체를 사용할 수 있다.

    "kind": "absolute | calendar",

    kind별 의미:
      -absolute:절대 시점을 의미한다. value에 yyyyMMddHHmm 형식의 숫자를 사용한다.
      {
        "kind": "absolute",
        "value": 202601011230
      }
      -calendar: 상대적인 달력 기준 시점을 의미한다. unit / offset / position을 사용한다.
      -unit: 기준 단위 (day, week, month, year)
      -offset: 기준 시점으로부터의 이동 값
      -position: 해당 단위의 시작(start) 또는 끝 (end)

    - kind가 absolute인 경우 value만 사용하며,
      unit / offset / position은 반드시 null이다.

    - kind가 calendar인 경우 unit / offset / position만 사용하며,
      value는 반드시 null이다.

    calendar kind의 기준 시점(now)은
    필터를 해석하는 서버 시점이며,
    LLM은 기준 시점을 계산하거나 추론하지 않는다.

    의미 예시:
    - "이번 주 말까지" → to: { kind: calendar, unit: week, offset: 0, position: end }
    - "다음 달 초부터" → from: { kind: calendar, unit: month, offset: 1, position: start }
    - "최근 7일 이내" → from: { kind: calendar, unit: day, offset: -7, position: null }
    값을 해석하거나 날짜로 변환하지 않는다.

    -"마감 임박" 같은 표현의 경우 마감일이 현재로부터 3일 이내인 공고를 조회하는 것과 동일하다.
    - "접수 중", "진행 중", "마감 전", "아직 마감되지 않은" 등의 표현은 endDate가 현재 시점 이후인 공고를 의미한다. 
      따라서 base는 반드시 endDate로 설정하고, from은 calendar offset=0, position=null 로 설정한다. to는 null로 설정한다.
    --------------------------------------------------

    연산자(op) 의미
    - "gte": 이상 (>=)
    - "gt": 초과 (>)
    - "lte": 이하 (<=)
    - "lt": 미만 (<)
    - "eq": 동일 (=)

    --------------------------------------------------

    금액 단위(억, 만 등)가 포함된 경우
    LLM은 이를 원 단위 숫자로 변환하여 value에 선언한다.
    -“기초금액 10억 이상” → 1000000000

    비율 관련 필드(minimumBidRate)는
    항상 0과 100 사이의 실수 값으로 표현한다.

    - 사용자가 "%", "퍼센트", "이상", "이하"와 함께
      1 이상 숫자를 언급한 경우:
      → 해당 값은 이미 퍼센트로 표현된 값으로 간주하고 그대로 사용한다.

      예:
      "낙찰하한율 88.5 이상" → 88.5
      "낙찰하한율 90 이하" → 90

    - 사용자가 1 이하의 소수를 직접 언급한 경우:
      → 비율로 표현된 값으로 간주하고 100을 곱한다.

      예:
      "낙찰하한율 0.12 미만" → 12

    수치 관련 필드에서
    "이상", "초과" → from 사용
    "이하", "미만" → to 사용

    from.op 은 반드시 "gte" 또는 "gt" 만 허용한다.
    단, 단일 값 비교의 경우(from만 사용하는 경우)에 한해
    from.op = eq 를 예외적으로 허용한다.
    to.op   는 반드시 "lte" 또는 "lt" 만 허용한다.

    "A 이상 B 이하"와 같이 범위가 명시된 경우:
    - 반드시 from과 to를 모두 설정한다.
    - 하나의 조건으로 축약하지 않는다.

    "A" 하나만 명시된 경우:
    - eq가 아닌 경우, 방향에 따라 from 또는 to만 설정한다.

    "~같", "~인", "~인 공고" 등
    단일 값 비교의 경우:

    - from과 to를 모두 사용하지 않고
    - from.op = eq 만 사용하며 to 는 반드시 null 이다.

    --------------------------------------------------
    output은 사용자가 원하는 결과 형태를 선언한다.

    output은 항상 배열이며 최소 1개 항목을 가진다.

    output 항목은 아래 4가지 type 중 하나이다.

    1) type="all"
      - 공고 전체 row를 조회한다.
      - field=null, op=null 이어야 한다.

    2) type="field"
      - 특정 필드 값만 조회한다.
      - field는 반드시 지정해야 한다.
      - op는 반드시 null이다.

    3) type="agg"
      - 집계 결과를 조회한다.
      - op는 반드시 지정해야 한다.
      - field는 count를 제외하고 반드시 지정해야 한다.
      - count 집계의 경우 field=null 허용한다.
    4) type="error"
      - 요청이 모호하거나 output 충돌이 발생한 경우 사용한다.
      - field=null, op=null 이어야 한다.
      - 서버는 type="error"가 포함되면 절대 DB 쿼리를 실행하지 않는다.

    - output에는 agg를 여러 개 포함할 수 있다.
    - output에 agg가 없으면 일반 조회를 실행한다.
    - output에는 agg와 field/all을 절대 혼합하지 않는다.
    - output 배열에 agg와 field/all이 동시에 존재하면 잘못된 요청이다. 이 경우 output은 [{"type":"error","field":null,"op":null}] 로 설정한다.
    -type="field"인 경우 op는 반드시 null이며 절대 다른 값을 넣지 않는다.
    - "개수", "몇 개", "총 몇 건", "count" 등의 집계를 명시적으로 요구한 경우에만 type="agg"를 사용한다.
    - "평균", "합계", "최대", "최소" 등의 집계 연산이 명시된 경우에만 type="agg"를 사용한다.
    - 그 외의 모든 "보여줘", "알려줘", "조회해줘" 등의 표현은   기본적으로 row 조회(type="all")로 처리한다.
    - 조건이 포함되어 있더라도 개수/집계를 명시하지 않은 경우 절대 type="agg"를 사용하지 않는다.
    --------------------------------------------------

    limit 규칙:

    - limit은 row 조회(all/field)일 때만 의미가 있다.
    - output에 agg만 있는 경우 limit은 반드시 null이다.
    - 집계(count, avg, sum 등) 질문에는 limit을 절대 포함하지 않는다.
    - 사용자가 단일 공고를 특정하면 limit=1로 설정한다.
    - row 조회(all/field)이며 사용자가 개수 제한을 명시하지 않은 경우, limit은 3으로 설정한다.
    - output이 error인 경우 limit은 반드시 null이다.

    error 처리 규칙:
    아래 조건에 만족하면 output은 반드시 [{"type":"error","field":null,"op":null}] 로 설정한다.
    [질문이 애매모호한 경우]
    - 사용자의 질문이 공고 목록 조회 의도를 명확히 표현했더라도,
      조회 대상이 모호하거나 조건 해석에 추론이 필요한 경우
    - "최근 공고", "적당한 공고", "괜찮은 공고"처럼
      구체적인 조건이 없는 경우
    
    [조건 충돌 규칙]
    - 조회 의도가 명확하더라도 필터 조건이 해석 불가능하거나 다의적인 경우
    - 동일 필드에 대해 상충되는 조건이 동시에 존재하면 output은 반드시 error로 설정한다.

      예:
      - "서울이고 대전인 공고"
      - "기초금액 10억 이상 5억 이하"
      - "마감일이 2026년 이후이고 2025년 이전"

    - 하나의 필드에 대해 서로 다른 값이 논리적으로 동시에 성립할 수 없는 경우
    - 서로 다른 base를 가지는 timeRange를 동시에 생성할 수 없다.

    [정의되지 않은 필드 처리 규칙]
    - 사용자의 표현이 스키마에 존재하지 않는 필드에 해당하면 의미가 유사하더라도 다른 필드로 절대 매핑하지 않는다.

    - 예:
      "지원금" ≠ estimatePrice
      "사업비" ≠ basicPrice
      "예산" ≠ estimatePrice
      "공사 규모" ≠ estimatePrice
      "등급" ≠ bidRange

    - 금액이 함께 언급되었더라도
      정의되지 않은 필드명과 결합된 수치 조건은
      절대 기존 금액 필드로 매핑하지 않는다.

      [estimatePrice 허용 표현]
      - 추정금액
      - 추정 가격
      - 추정가
      - 예상금액
      - 예상 가격

      [basicPrice 허용 표현]
      - 기초금액
      - 기초 가격
      - 기초가

      - 위 허용 표현과 띄어쓰기 차이, 조사 결합, 단순 어미 변화는 허용한다. (예: "추정 가격이", "기초금액은", "예상 금액의")
      - 정의된 동의어 그룹에 포함된 표현은 해당 필드로 매핑한다.
      - 동의어 그룹에 포함되지 않은 표현은 의미가 유사하더라도 절대 매핑하지 않는다.
      - 정의되지 않은 구조적 필드가 단독 조건이거나 수치 조건과 결합된 경우에만 error로 처리한다.
      - 다른 정상 필드와 함께 존재하더라도, 해당 정의되지 않은 필드가 구조적 의미를 가진 경우 전체 요청은 error로 처리한다.
      - 사용자의 질문에 포함된 조건을 임의로 제거하거나 무시하지 않는다.
      - 정의되지 않은 구조적 필드가 포함된 경우, 해당 조건을 삭제하고 나머지 조건만으로 재해석하지 않는다.
      - 정의되지 않은 구조적 필드가 포함되면 전체 요청을 반드시 error로 처리한다.

    

    필드 의미 참고:

    - bidRealId: 공고번호
    - name: 공고명
    - region: 지역
    - organization: 발주기관
    - estimatePrice: 추정가격
    - basicPrice: 기초금액
    - minimumBidRate: 낙찰하한율(%)
    - bidRange: 투찰률
    - startDate: 공고 시작일
    - endDate: 공고 마감일
    - openDate: 개찰일

    사용 가능한 필드:
    {
    "limit": number | null,
    "filter": {
      "bidRealId": string | null,
      "name": string | null,
      "region": string | null,
      "organization": string | null,

    "estimatePrice": {
      "from": { "op": "gte" | "gt" | "eq", "value": number } | null,
      "to":   { "op": "lte" | "lt",        "value": number } | null
    } | null,

    "basicPrice": {
      "from": { "op": "gte" | "gt" | "eq", "value": number } | null,
      "to":   { "op": "lte" | "lt",        "value": number } | null
    } | null,

    "minimumBidRate": {
      "from": { "op": "gte" | "gt" | "eq", "value": number } | null,
      "to":   { "op": "lte" | "lt",        "value": number } | null
    } | null,

    "bidRange": {
      "from": { "op": "gte" | "gt" | "eq", "value": number } | null,
      "to":   { "op": "lte" | "lt",        "value": number } | null
    } | null,

    "timeRange": {
      "base": "startDate" | "endDate" | "openDate",
      "from": {
        "kind": "absolute" | "calendar",
        "value": number | null,
        "unit": "day" | "week" | "month" | "year" | null,
        "offset": number | null,
        "position": "start" | "end" | null
      },
      "to": {
        "kind": "absolute" | "calendar",
        "value": number | null,
        "unit": "day" | "week" | "month" | "year" | null,
        "offset": number | null,
        "position": "start" | "end" | null
      }
    } | null
  },
  "output": [
  {
    "type": "all" | "field" | "agg" | "error",
    "field": "bidRealId" | "name" | "region" | "organization" | "estimatePrice" | "basicPrice" | "minimumBidRate" | "bidRange" | "startDate" | "endDate" | "openDate" | null,
    "op": "min" | "max" | "avg" | "sum" | "count" | null
  }
]
}

    구조화 결과 예시:

    예시 1)
    입력: "공고번호 20240123456-000 내용 알려줘"

    출력:
    {
      "limit": 1,
      "filter": {
        "bidRealId": "20240123456-000",
        "name": null,
        "region": null,
        "organization": null,
        "estimatePrice": null,
        "basicPrice": null,
        "minimumBidRate": null,
        "bidRange": null,
        "timeRange": null
      },
      "output": [{"type": "all", "field": null, "op":null}]
    }

    예시 2)
    입력: "부산 지역에 2026년 1월 1일부터 다음 달 말까지 개찰일 기준 에이블스쿨에서 진행하는 학교 급식실 공사 공고의 개수를 보여줘"

    출력:
    {
      "limit": null,
      "filter": {
        "bidRealId": null,
        "name": "학교 급식실",
        "region": "부산",
        "organization": "에이블스쿨",
        "estimatePrice": null,
        "basicPrice": null,
        "minimumBidRate": null,
        "bidRange": null,
        "timeRange": {
          "base": "openDate",
          "from": {
            "kind": "absolute",
            "value": 202601010000,
            "unit": null,
            "offset": null,
            "position": null
          },
          "to": {
            "kind": "calendar",
            "value": null,
            "unit": "month",
            "offset": 1,
            "position": "end"
              }
            }
      },
      "output": [
      {"type": "agg", "field": null, "op": "count"}
      ]
    }


    예시 3)
    입력: "마감일이 202601010000부터 202601011234인 서울에서 기초금액 10억 이상이고 낙찰하한율이 0.8 이상이고 95%보다 작은 공고의 기초금액 평균을 알려줘"

    출력:
    {
      "limit": null,
      "filter": {
        "bidRealId": null,
        "name": null,
        "region": "서울",
        "organization": null,
        "estimatePrice": null,
        "basicPrice": {
          "from":{"op": "gte", "value": 1000000000},
          "to": null
        },
        "minimumBidRate":{
            "from":{"op": "gte", "value": 80},
            "to":{"op": "lt", "value":95}
          },
        "bidRange": null,
        "timeRange": {
        "base": "endDate",
        "from": {
          "kind": "absolute",
          "value": 202601010000,
          "unit": null,
          "offset": null,
          "position": null
          },
        "to": {
          "kind": "absolute",
          "value": 202601011234,
          "unit": null,
          "offset": null,
          "position": null
        }
        }
        },
        "output": [
        {"type" : "agg", "field" : "basicPrice", "op" : "avg"}
        ]
    }

    예시 4) 
    입력: "2023년11월11일부터 내년초까지 공고 중 공고의 마감일과 추정가격을 알려줘"

    출력: 
    {
      "limit": 3,
      "filter": {
        "bidRealId": null,
        "name": null,
        "region": null,
        "organization": null,
        "estimatePrice": null,
        "basicPrice": null,
        "minimumBidRate": null,
        "bidRange": null,
        "timeRange": {
          "base": "endDate",
          "from": {
            "kind": "absolute",
            "value": 202311110000,
            "unit": null,
            "offset": null,
            "position": null
          },
          "to": {
            "kind": "calendar",
            "value": null,
            "unit": "year",
            "offset": 1,
            "position": "start"
              }
            }
      }
      },
      "output": [
        { "type": "field", "field": "endDate", "op": null },
        { "type": "field", "field": "estimatePrice", "op": null }
      ]
    }
    """ 

    #프롬프트 짧게 짰을 때 성능 확인용
    prompt_kor="""
    당신은 엄격한 JSON 생성기입니다.

    당신의 임무는 사용자의 자연어 질문을 제공된 스키마에 엄격히 맞는 JSON 객체로 변환하는 것입니다.

    다음 규칙을 반드시 따라야 합니다:

    1. 출력 형식
      - 출력은 단일 유효한 JSON 객체여야 합니다.
      - 설명, 주석, 마크다운을 포함하지 마세요.
      - JSON 외의 텍스트는 절대 포함하지 마세요.

    2. 스키마 준수
      - JSON 구조는 스키마와 정확히 일치해야 합니다.
      - 새로운 필드를 추가하지 마세요.
      - 필수 필드를 생략하지 마세요.
      - 확신이 없는 경우 값은 반드시 null로 설정하세요.

    3. 허구 금지
      - 사용자가 명시적으로 말하거나 강하게 암시한 것만 추출하세요.
      - 추측하지 마세요.
      - 조건이 모호하거나 충분히 명시되지 않은 경우 null로 설정하세요.
      - region과 organization 해석은 아래 규칙을 따르세요.

    4. 필터
      - 모든 검색 조건은 "filter" 아래에 포함되어야 합니다.
      - 여러 조건을 동시에 적용할 수 있습니다.
      - 필터 조건이 없는 경우 "filter"를 null로 설정하세요.

    5. 범위 필드 (estimatePrice, basicPrice, minimumBidRate, bidRange)
      - 사용자가 명시적으로 범위를 암시한 경우에만 "from"과/또는 "to"를 사용하세요.
      - 연산자 선택은 오직 문장 표현에 따라 gte / gt / eq / lte / lt로 제한합니다.
      - 범위의 한쪽만 언급된 경우, 다른 쪽은 null로 설정하세요.

    6. 시간 범위 처리
      - 모든 날짜 관련 표현은 "timeRange"로 표현해야 합니다.
      - "base"는 startDate, endDate, openDate 중 하나여야 합니다.
      - 구체적인 타임스탬프를 지정한 경우 "absolute"를 사용하세요.
      - 상대적 표현 (예: last month, this week)에는 "calendar"를 사용하세요.
      - 시작 또는 끝만 언급된 경우, 다른 쪽은 null로 설정하세요.
      - 날짜 참조가 불분명하면 "timeRange"를 null로 설정하세요.

    7. 출력 필드
      - 사용자가 전체 레코드를 원하거나 출력 필드를 지정하지 않은 경우:
        { "type": "all", "field": null, "op": null }를 사용하세요.
      - 특정 필드를 요청하면 type = "field"를 사용하세요.
      - 집계(count, min, max, avg, sum)를 요청하면 type = "agg"를 사용하세요.
      - 데이터베이스 쿼리로 실행할 수 없는 경우(명확한 대상 없음, 출력 유형 충돌, 비쿼리 의도 등)는 output.type = "error"로 설정하세요.
      - 필터 없이 단순 조회 의도(예: "show anything", "recent notices")는 유효한 쿼리로 간주하며 error로 처리하지 마세요.

    8. Limit
      - 사용자가 개수를 명시적으로 지정한 경우에만 limit을 설정하세요.
      - 단순 조회 의도에서 개수를 지정하지 않은 경우 limit은 null로 두어도 되며, 서버에서 기본 제한을 적용할 수 있습니다.

    9. 낮은 신뢰도 처리
      - 잘못된 추론보다는 null을 선호하세요.
      - 대부분 null로 이루어진 JSON 객체를 반환하는 것도 허용됩니다.

    10. 지역 vs 기관 해석 (한국어 규칙)
      - 표현이 명확하게 위치나 장소를 나타내면 filter.region으로 매핑하세요.
        예:
          - "서울 공고"
          - "서울에서 한 공사"
          - "부산 지역 공사"
        → region = "<location>", organization = null
      - 표현이 명확하게 발주 기관이나 기관을 나타내면 filter.organization으로 매핑하세요.
        예:
          - "서울시 공고"
          - "국토교통부 공사"
          - "조달청 발주"
        → organization = "<institution>", region = null
      - 위치와 기관이 모두 명시된 경우 둘 다 설정할 수 있습니다.
      - 위치 이름만으로 기관을 추론하지 마세요.

    11. 출력 일관성 규칙
      - "output" 필드는 여러 결과를 허용하기 위해 배열이어야 합니다.
      - 모든 객체는 동일한 "type"을 가져야 합니다.
      - 서로 다른 출력 유형(all과 agg 등)을 혼합하지 마세요.
      - 여러 집계 결과를 요청하는 경우, output 배열에 여러 "agg" 객체를 사용하세요.
      - output.type = "all"이면 배열에 정확히 하나의 객체만 포함해야 합니다.
      - output.type = "field"이면 배열에 여러 객체를 포함할 수 있지만, 모두 type = "field"여야 합니다.
      - 그룹화 쿼리 또는 field+agg 혼합 쿼리는 지원되지 않습니다. 요청 시 output.type = "error"로 설정하세요.

    마지막으로 너는 이 json 스키마에 맞춰서 json을 생성해야한다.

    {
    "limit": number | null,
    "filter": {
      "bidRealId": string | null,
      "name": string | null,
      "region": string | null,
      "organization": string | null,


    "estimatePrice": {
      "from": { "op": "gte" | "gt" | "eq", "value": number } | null,
      "to":   { "op": "lte" | "lt",        "value": number } | null
    } | null,


    "basicPrice": {
      "from": { "op": "gte" | "gt" | "eq", "value": number } | null,
      "to":   { "op": "lte" | "lt",        "value": number } | null
    } | null,


    "minimumBidRate": {
      "from": { "op": "gte" | "gt" | "eq", "value": number } | null,
      "to":   { "op": "lte" | "lt",        "value": number } | null
    } | null,


    "bidRange": {
      "from": { "op": "gte" | "gt" | "eq", "value": number } | null,
      "to":   { "op": "lte" | "lt",        "value": number } | null
    } | null,


    "timeRange": {
      "base": "startDate" | "endDate" | "openDate",
      "from": {
        "kind": "absolute" | "calendar",
        "value": number | null,
        "unit": "day" | "week" | "month" | "year" | null,
        "offset": number | null,
        "position": "start" | "end" | null
      },
      "to": {
        "kind": "absolute" | "calendar",
        "value": number | null,
        "unit": "day" | "week" | "month" | "year" | null,
        "offset": number | null,
        "position": "start" | "end" | null
      }
    } | null
  },
  "output": [
  {
    "type": "all" | "field" | "agg" | "error",
    "field": "bidRealId" | "name" | "region" | "organization" | "estimatePrice" | "basicPrice" | "minimumBidRate" | "bidRange" | "startDate" | "endDate" | "openDate" | null,
    "op": "min" | "max" | "avg" | "sum" | "count" | null
  }
]
}

    }
"""
    prompt_eng="""
    You are a strict JSON generator.

Your task is to convert a user's natural language query into a JSON object
that strictly conforms to the provided schema.

You MUST follow these rules:

1. Output format
   - Output must be a single valid JSON object.
   - Do NOT include explanations, comments, or markdown.
   - Do NOT include any text outside the JSON.

2. Schema compliance
   - The JSON structure MUST exactly match the schema.
   - Do NOT add new fields.
   - Do NOT omit required fields.
   - Use null explicitly when a value cannot be confidently determined.

3. No hallucination
   - Extract only what the user explicitly states or strongly implies.
   - Do NOT guess.
   - If a condition is ambiguous or underspecified, set it to null.
   - When interpreting region vs organization, follow the rules defined below.
   
4. Filters
   - All search conditions belong under "filter".
   - Multiple conditions may be applied simultaneously.
   - If no filtering condition is specified, set "filter" to null.

5. Range fields (estimatePrice, basicPrice, minimumBidRate, bidRange)
   - Use "from" and/or "to" ONLY when the user explicitly implies a range.
   - Choose operators strictly based on wording: gte / gt / eq / lte / lt
   - If only one side of the range is mentioned, the other side must be null.

6. Time range handling:
   - All date-related expressions MUST be represented using "timeRange".
   - "base" must be one of: startDate, endDate, openDate.
   - Use "absolute" when the user specifies a concrete timestamp.
   - Use "calendar" when the user uses relative expressions (e.g. last month, this week).
   - If only a start or end is mentioned, the other side must be null.
   - If the date reference is unclear, set "timeRange" to null.

7. Output field:
   - If the user wants full records or does not specify output fields, use: { "type": "all", "field": null, "op": null }
   - If the user asks for specific fields, use type = "field".
   - If the user asks for aggregation (count, min, max, avg, sum), use type = "agg".
   - If the user's query cannot be executed as a database query
    (e.g. no clear target, conflicting output types, or non-query intent), set output.type = "error".
  -Browsing intent without filters (e.g. "show anything", "any notice", "recent notices") is considered a valid query and MUST NOT be treated as an error.

8. Limit:
   - Set "limit" only if the user explicitly specifies a count.
   - If the user expresses browsing intent without specifying a count, limit may remain null. A default limit may be applied by the server.

9. Low-confidence handling
   - Prefer null over incorrect inference.
   - It is always acceptable to return a mostly-null JSON object.

10. Region vs Organization interpretation (Korean language rules)
   - If the expression clearly indicates a physical location or place,
     map it to filter.region.
     Examples:
       - "서울 공고"
       - "서울에서 한 공사"
       - "부산 지역 공사"
     → region = "<location>", organization = null
   - If the expression clearly indicates an issuing authority or institution,
     map it to filter.organization.
     Examples:
       - "서울시 공고"
       - "국토교통부 공사"
       - "조달청 발주"
     → organization = "<institution>", region = null
   - If BOTH a location and an institution are explicitly mentioned,
     it is allowed to set BOTH region and organization.
   - Do NOT infer organization from a location name alone.

  11. Output consistency rules
   - The "output" field MUST be an array to allow multiple results.
   - All objects inside "output" MUST have the same "type".
   - Mixing different output types (e.g. "all" with "agg") is NOT allowed.
   - If multiple aggregation results are requested, use multiple "agg" objects inside the output array.
   - If output.type = "all", the output array MUST contain exactly one object.
   - If output.type = "field", the output array may contain multiple objects, but all MUST have type = "field".
   - Group-by queries or field+agg mixed queries are NOT supported.
     If the user requests them, set output.type = "error".
You must strictly follow this schema:

    {
    "limit": number | null,
    "filter": {
      "bidRealId": string | null,
      "name": string | null,
      "region": string | null,
      "organization": string | null,


    "estimatePrice": {
      "from": { "op": "gte" | "gt" | "eq", "value": number } | null,
      "to":   { "op": "lte" | "lt",        "value": number } | null
    } | null,


    "basicPrice": {
      "from": { "op": "gte" | "gt" | "eq", "value": number } | null,
      "to":   { "op": "lte" | "lt",        "value": number } | null
    } | null,


    "minimumBidRate": {
      "from": { "op": "gte" | "gt" | "eq", "value": number } | null,
      "to":   { "op": "lte" | "lt",        "value": number } | null
    } | null,


    "bidRange": {
      "from": { "op": "gte" | "gt" | "eq", "value": number } | null,
      "to":   { "op": "lte" | "lt",        "value": number } | null
    } | null,


    "timeRange": {
      "base": "startDate" | "endDate" | "openDate",
      "from": {
        "kind": "absolute" | "calendar",
        "value": number | null,
        "unit": "day" | "week" | "month" | "year" | null,
        "offset": number | null,
        "position": "start" | "end" | null
      },
      "to": {
        "kind": "absolute" | "calendar",
        "value": number | null,
        "unit": "day" | "week" | "month" | "year" | null,
        "offset": number | null,
        "position": "start" | "end" | null
      }
    } | null
  },
  "output": [
  {
    "type": "all" | "field" | "agg" | "error",
    "field": "bidRealId" | "name" | "region" | "organization" | "estimatePrice" | "basicPrice" | "minimumBidRate" | "bidRange" | "startDate" | "endDate" | "openDate" | null,
    "op": "min" | "max" | "avg" | "sum" | "count" | null
  }
]
}

    }"""
    '''
    prompt=prompt_pydantic_kor+f"""
    질문:
    {user_query}
    
    출력은 json 객체 하나만 반환해라.
    """
    '''
    messages=[SystemMessage(content=prompt), #promt: 긴 한국어 / prompt_kor: 짧은 한국어 / prompt_eng: 짧은 영어
    HumanMessage(content=user_query)]
    #아래 주석은 응답시간 테스트용
    #import time
    #start=time.time()
    response = llm.invoke(messages)
    #end=time.time()
    #print("응답시간:",end-start)
    
    raw = response.content.strip()

    try:
      raw_json = json.loads(raw)
    except json.JSONDecodeError:
        # JSON이 아니면 error 반환
        return {
            "limit": None,
            "filter": None,
            "output": [{"type": "error", "field": None, "op": None}]
        }
  

    try:
        parsed = Query.model_validate(raw_json)
        result_dict = parsed.model_dump(by_alias=True)

        pretty_json_string = json.dumps(
            result_dict,
            indent=2,
            ensure_ascii=False
        )

        # 문자열 그대로 반환
        return pretty_json_string
    except Exception as e:
      print("❌ Pydantic error:", e)
      print("❌ Raw JSON:", raw_json)
      error_json = {
                "__error__": "pydantic_validation",
                "__raw__": raw_json
            }
      return error_json
    #return response.content