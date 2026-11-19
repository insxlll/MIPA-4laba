# Документація промптів

Усі промпти асистента зібрані тут у тому вигляді, в якому вони використовуються в `ai_analyst_assistant.ipynb`. Для кожного описано, навіщо така структура, що пробував до цього і як змінював.

## Загальні принципи (для всіх промптів)

1. **LLM не рахує числа.** Усі метрики рахує Pandas, а модель отримує вже пораховані факти. LLM відповідає лише за інтерпретацію, формулювання і код. Це головний захист від галюцинацій.
2. **Структурований вихід (JSON mode).** У кожному виклику задано `responseMimeType: application/json`, а в промпті є точна схема відповіді. JSON легко розібрати і перевірити програмно, на відміну від вільного тексту.
3. **Системна інструкція окремо від даних.** Правила лежать у `systemInstruction`, а дані й питання користувача в `contents`. Так правила не губляться серед великої таблиці фактів.
4. **Промпти англійською, відповіді українською.** Інструкції англійською модель виконує стабільніше. Мова відповіді задається окремим правилом `Write all text in Ukrainian`.
5. **Явні правила у формі списку.** Короткі правила («Use ONLY numbers…», «Cite ONLY existing fact_ids») модель порушує рідше, ніж правила, сховані в довгому абзаці.
6. **Температура 0.2.** Для аналітики потрібні повторювані результати, а не креативність.
7. **Програмна перевірка після кожного виклику.** Навіть із правилами модель може помилитися, тому код перевіряє колонки, сегменти, `fact_id` і числа (`check_numbers`).

---

## 1. Автоматичний EDA: `EDA_SYSTEM`

```text
You are a senior product data analyst reviewing data quality.
You receive a JSON profile of ONE table computed with pandas. You never see raw rows.
Rules:
- Use ONLY numbers that appear in the profile. Never invent or recompute numbers.
- Every issue must reference an existing column name from the profile.
- Be specific: say which column, what is wrong, and the evidence from the profile.
- Severity: "high" breaks metrics, "medium" may bias results, "low" is cosmetic.
- Write all text in Ukrainian.
Return JSON exactly in this schema:
{"summary": "2-3 sentences about the table and its overall quality",
 "issues": [{"column": "...", "problem": "...", "evidence": "...", "severity": "high|medium|low", "fix": "..."}],
 "observations": ["notable facts about distributions, 3-5 items"],
 "next_steps": ["what an analyst should check next, 2-4 items"]}
```

**Що отримує модель:** не сирі рядки, а JSON-профіль таблиці від `profile_table()`: тип, пропуски, унікальні значення, квантилі, викиди 3×IQR, від'ємні й нульові значення, діапазон дат, пробіли й різний регістр у тексті.

**Чому така структура:**
- `You never see raw rows` пояснює моделі, що вона не може перевірити дані сама, тож має спиратися лише на профіль.
- Поле `evidence` у кожній проблемі змушує модель вказати доказ із профілю, а не загальну фразу «можливі пропуски».
- `severity` зі строгим визначенням (`high` ламає метрики, `medium` може зсунути, `low` косметика) дає однакову шкалу між запусками.

**Як дійшов до цієї версії:**
- *Найпростіший варіант:* «Проаналізуй цю таблицю» + перші 100 рядків CSV. Відмовився: модель бачила б лише вибірку (у `registrations` 2 пропуски віку на 202 тис. рядків у перші 100 рядків не потрапляють) і була б змушена «вгадувати» середні та діапазони.
- *Другий варіант:* профіль замість сирих даних, але відповідь вільним текстом. Відмовився: зі звичайного тексту неможливо надійно витягнути список проблем і перевірити, на які колонки вони посилаються.
- *Поточна версія:* JSON-схема з `issues[]`, `column`, `evidence`. Код відкидає проблеми з неіснуючими колонками і звіряє числа з профілем. Обидві перевірки протестовано на навмисно хибних відповідях (вигадана колонка `foo`, вигадане число): вони відсіюються.

---

## 2. Генерація гіпотез: `HYPOTHESES_SYSTEM`

```text
You are a senior product analyst. You receive a table of facts computed with pandas
about a freemium mobile app (users, sessions, payments). Each fact has a fact_id.
Task: propose {n} hypotheses worth investigating.
Rules:
- Each hypothesis must be supported by 1-3 fact_ids from the table. Cite ONLY existing fact_ids.
- Do not write numbers yourself: the system inserts the values for the cited fact_ids.
- Compare a segment with its baseline (fact rows with segment "ALL") or with other segments.
- Prefer big, actionable differences; ignore segments with fewer than 500 users.
- Mention possible data-quality explanations when a pattern looks like a tracking artifact.
- Write all text in Ukrainian.
Return JSON: {"hypotheses": [{"title": "short", "hypothesis": "1-2 sentences",
 "fact_ids": ["F001"], "why_it_matters": "...", "how_to_check": "concrete analysis to run",
 "priority": "high|medium|low"}]}
```

**Що отримує модель:** таблицю фактів від `build_fact_table()` у форматі CSV: `fact_id, dimension, segment, metric, value, n_users`. Метрики: кількість користувачів, конверсія в оплату, ARPU, сесії й хвилини на користувача в розрізах платформи, статі, віку, країни, нові/старі; реєстрації й конверсія по тижнях; середній DAU за днями тижня.

**Чому така структура:**
- Головне рішення: **модель посилається на `fact_id`, а не пише числа**. Значення підставляються з таблиці фактів кодом, тому вигадати число в доказах модель фізично не може. Гіпотези з неіснуючими `fact_id` відкидаються.
- Рядок `segment = "ALL"` дає базу для порівняння, тому модель формулює гіпотезу як відхилення сегмента від середнього.
- `ignore segments with fewer than 500 users` прибирає «гіпотези» на шумі малих вибірок (країни з 20 користувачами).
- `Mention possible data-quality explanations` з'явилося, бо в даних є артефакт трекінгу (стрибок DAU 19.04, коли в sessions з'являються старі користувачі). Без цього правила модель трактувала б його як реальний ріст.
- `how_to_check` змушує дати конкретний наступний крок аналізу, а не «варто дослідити».

**Як дійшов до цієї версії:**
- *Найпростіший варіант:* «Запропонуй 10 гіпотез» + лише опис колонок. Відмовився: без порахованих метрик модель може видати тільки загальні фрази без цифр, а завдання вимагає гіпотез, обґрунтованих числами з датасету.
- *Другий варіант:* дати метрики й дозволити моделі писати числа в текст. Відмовився: кожне число довелося б перевіряти вручну, а таблиця має 274 факти, тож переплутати сегменти легко.
- *Поточна версія:* посилання на `fact_id` + підстановка значень кодом + фільтр малих сегментів. Протестовано на навмисно хибній відповіді з неіснуючим `F999`: така гіпотеза відкидається.

---

## 3. Natural Language → Pandas: `NL2PANDAS_SYSTEM` і `INTERPRET_SYSTEM`

```text
You translate analytics questions into pandas code.
Available pandas DataFrames (already loaded) and their schema are given below.
Rules:
- Use only pandas (pd), numpy (np) and the listed DataFrames. No imports, no file or network access, no plotting.
- Do not modify the DataFrames in place; build new objects.
- Store the final answer in a variable named result (a DataFrame, Series or scalar). Keep it small: at most 50 rows.
- Use the exact column names from the schema. Dates are pandas datetime64.
- If the question cannot be answered with these tables (or is not about the data), set can_answer to false and explain why.
- Write explanation and assumptions in Ukrainian.
Return JSON: {"can_answer": true, "code": "...", "explanation": "what the code does, 1-2 sentences", "assumptions": ["..."]}

Schema:
{schema}
```

Замість `{schema}` підставляється схема всіх таблиць: назва, кількість рядків, колонки з типами і 3 приклади рядків.

```text
You are a product analyst. Given a question and the computed result (from pandas),
write a short answer in Ukrainian (2-3 sentences). Use only numbers present in the result. Return JSON: {"answer": "..."}
```

**Чому така структура:**
- **Схема з прикладами рядків**, а не лише назви колонок: модель бачить формат значень (наприклад, `app` = `mobile/ios/...`, дати як datetime) і рідше помиляється у фільтрах.
- **Одна змінна `result`** дає простий і передбачуваний контракт між моделлю і кодом, який виконує відповідь.
- `can_answer: false` дає моделі законний спосіб відмовитись (питання про погоду, про дані, яких немає), замість того щоб вигадувати код.
- `No imports, no file or network access` дублюється **AST-перевіркою** в коді: навіть якщо модель порушить правило, небезпечний код не виконається. Виконання йде на копіях таблиць з обмеженим набором вбудованих функцій.
- **Два кроки** (код → інтерпретація) замість одного: числа у відповіді беруться з реального результату виконання, а не з «уявлення» моделі про дані.

**Як дійшов до цієї версії:**
- *Найпростіший варіант:* код і пояснення одним текстом. Відмовився: код довелося б вирізати регулярками з markdown, а це ламається на першому ж нестандартному форматуванні.
- *Другий варіант:* JSON з полем `code`, але без механізму виправлення. Недолік: одна помилка в назві колонки (наприклад, `platform` замість `app`) зупиняє весь сценарій.
- *Поточна версія:* схема з прикладами рядків + **автовиправлення**: якщо код падає, модель отримує текст помилки й повертає виправлений код (один повтор). Плюс `can_answer` і AST-перевірка безпеки. Протестовано: код з `KeyError: 'platform'` виправляється з другої спроби; код з `import os` блокується до виконання; питання про погоду отримує відмову.

---

## 4. Пояснення зміни метрики: `CHANGE_SYSTEM`

```text
You are a product analyst explaining why a metric changed between two periods.
You receive a pandas decomposition: total change and the segments with the largest contribution.
Rules:
- Use only numbers from the input. Do not invent segments that are not in the input.
- Distinguish real business change from artifacts (unequal period length, tracking changes, data cut-off).
- Write in Ukrainian.
Return JSON: {"headline": "one sentence", "drivers": [{"dimension": "...", "segment": "...", "explanation": "..."}],
 "hypotheses": ["possible causes, 2-4"], "next_checks": ["what to verify, 2-3"]}
```

**Що отримує модель:** зведення (значення в обох періодах, кількість днів, зміна суми і зміна в середньому за день) і топ сегментів за внеском у зміну, пораховані `explain_change()`.

**Чому така структура:**
- Розклад по сегментах рахує код, тож модель не шукає «драйвери» навмання, а пояснює вже знайдені.
- `Distinguish real business change from artifacts` і поля `days_previous`/`days_current` потрібні, щоб модель не назвала падінням неповний тиждень. Код окремо попереджає про різну довжину періодів і порівнює середнє за день.
- Сегменти з відповіді звіряються зі списком реальних сегментів; вигадані позначаються ⚠️.

**Як дійшов до цієї версії:**
- *Перший варіант:* порівнювати суми за періоди. Перевірка на даних показала проблему: останній тиждень (6 днів, бо дані закінчуються 11.05) за сумою дає падіння виручки на 31,4%, і модель шукала б для нього «бізнес-причини».
- *Поточна версія:* нормалізація на день (реальна зміна -20,0%) + явне правило про артефакти + перевірка сегментів (вигаданий сегмент `app=tv` у тестовій відповіді позначається ⚠️).

---

## 5. Executive summary: `SUMMARY_SYSTEM`

```text
You write an executive summary for a product manager.
You receive KPIs computed with pandas for the current and previous period.
Rules:
- Use ONLY numbers from the KPI JSON (you may round them). Never compute new numbers.
- Business language, no jargon. Max 120 words in total.
- Structure: what happened -> why it matters -> what to do next.
- Write in Ukrainian.
Return JSON: {"headline": "one sentence", "what_happened": ["2-4 bullets"], "why_it_matters": "1-2 sentences", "next_steps": ["2-3 bullets"]}
```

**Що отримує модель:** KPI від `compute_kpis()` за поточний і попередній період (нові й активні користувачі, середній DAU, виручка, платники, середній чек, виручка за день) і відсоток зміни, а також топ-3 країни за виручкою.

**Чому така структура:**
- Структура «що сталось → чому важливо → що робити» стандартна для тексту стейкхолдеру; вона задана полями JSON, тому кожен блок завжди присутній.
- `Max 120 words` прибирає воду: продакт читає це за 30 секунд.
- `Use ONLY numbers from the KPI JSON (you may round them)` + **`check_numbers()`**: кожне число в тексті звіряється з KPI з урахуванням округлення і відсотків, а незнайдені позначаються ⚠️.

**Як дійшов до цієї версії:**
- Без обмеження довжини такий текст розростається до пів сторінки загальних порад, тому додано `Max 120 words` і жорстку структуру з трьох блоків.
- Відсотки зміни між періодами не довіряю рахувати моделі: їх заздалегідь рахує Pandas (`change_pct`), а модель лише цитує. Тестова відповідь із вигаданим «ростом на 999%» позначається перевіркою як ⚠️.

---

## Помилки, знайдені під час розробки й тестування

Логіку асистента протестовано прогоном усього notebook із підміненим API (навмисно хибні відповіді моделі), щоб перевірити кожну програмну перевірку. Під час цього знайдено й виправлено:

- `str.format()` на промпті з JSON-схемою падав з `KeyError`, бо фігурні дужки JSON сприймались як плейсхолдери. Підстановку замінено на `.replace('{n}', ...)`.
- Витягування чисел із тексту склеювало сусідні колонки таблиці через пробіл (`6.80   117.30`). Регулярний вираз переписано: пробіл вважається роздільником тисяч лише перед групою рівно з 3 цифр; дати перед перевіркою вирізаються.
- У довіднику країн є дубль коду (`NaN`), через що після `merge` таблиця `users` мала на один рядок більше, ніж `registrations`. Додано `drop_duplicates`.
