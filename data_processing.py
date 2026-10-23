"""
part 1: data extraction and cleaning for the brfss regression project.

reads the 2011-2014 brfss zip files one line at a time and yields clean
(predictors, health) observations for the regression code to consume.
"""

import os
import zipfile
from io import TextIOWrapper

def fieldDictBuild():
    fieldDict = dict.fromkeys([0, 1, 2, 3, 11, 12, 13, 14])

    fieldDict[11] = {
        "genhlth": 73,
        "bmi": (1533, 1536),
        "income": (124, 125),
        "education": 122,
    }
    fieldDict[12] = {
        "genhlth": 73,
        "bmi": (1644, 1647),
        "income": (116, 117),
        "education": 114,
    }
    fieldDict[13] = {
        "genhlth": 80,
        "bmi": (2192, 2195),
        "income": (152, 153),
        "education": 150,
    }
    fieldDict[14] = {
        "genhlth": 80,
        "bmi": (2247, 2250),
        "income": (152, 153),
        "education": 150,
    }

    return fieldDict

def getIncome(incomeString):
    if incomeString != "  ":
        income = int(incomeString)
    else:
        income = 9
    return income

def convertBMI(bmiString, shortYear):
    bmi = 0
    if shortYear == 0 and bmiString != "999":
        bmi = 0.1 * float(bmiString)
    if shortYear == 1 and bmiString != "999999":
        bmi = 0.0001 * float(bmiString)
    if 2 <= shortYear <= 10 and bmiString != "9999":
        bmi = 0.01 * float(bmiString)
    if shortYear > 10 and bmiString != "    ":
        bmi = 0.01 * float(bmiString)
    return bmi

def getEducation(educationString):
    if educationString != " ":
        education = int(educationString)
    else:
        education = 9
    return education

def getHlth(hlthString):
    if hlthString != " ":
        genhlth = int(hlthString)
        if genhlth > 6:
            genhlth = -1
    else:
        genhlth = -1

    assert genhlth in (-1, 1, 2, 3, 4, 5, 6)
    return genhlth


# ---------------------------------------------------------------------------
# part 1 code
# ---------------------------------------------------------------------------

VALID_COLUMNS = ("education", "income", "bmi")
YEARS = (11, 12, 13, 14)
FIELD_DICT = fieldDictBuild()


def zip_name(short_year):
    return f"LLCP20{short_year}ASC.ZIP"


def validate_columns(columns):
    if isinstance(columns, str) or not isinstance(columns, (list, tuple)):
        raise TypeError(f"columns must be a list like ['education', 'bmi'], got {columns!r}")
    if len(columns) == 0:
        raise ValueError("columns must contain at least one predictor")

    bad = [c for c in columns if c not in VALID_COLUMNS]
    if bad:
        raise ValueError(f"unsupported predictor(s) {bad}, choose from {list(VALID_COLUMNS)}")
    if len(set(columns)) != len(columns):
        raise ValueError(f"columns has duplicates: {list(columns)}")

    return list(columns)


def find_zip_files(data_dir):
    """return {short_year: path} for the four brfss zips, or fail loudly if any are missing."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data directory not found: {os.path.abspath(data_dir)}")

    # match names case-insensitively so llcp2011asc.zip works too
    on_disk = {name.upper(): name for name in os.listdir(data_dir)}

    paths = {}
    missing = []
    for year in YEARS:
        name = on_disk.get(zip_name(year))
        if name is None:
            missing.append(zip_name(year))
        else:
            paths[year] = os.path.join(data_dir, name)

    if missing:
        raise FileNotFoundError(
            f"missing brfss file(s) in {os.path.abspath(data_dir)}: {', '.join(missing)} "
            "(keep them zipped, do not extract)"
        )
    return paths


def is_valid(name, value):
    # same rules as step 10 of the professor's script
    if name == "education" or name == "income":
        return value < 9
    if name == "bmi":
        return 0 < value < 99
    if name == "genhlth":
        return value != -1


def parse_record(record, short_year, columns, require_all=False):
    """
    turn one fixed-width line into (predictors, health), or None if it should be skipped.

    the field positions in fieldDictBuild are 1-based and inclusive (matching the
    professor's template convention), so a field at position p is record[p - 1]
    and a range (s, e) is record[s - 1:e].

    returns None for records with invalid/missing values or parsing errors.
    """
    fields = FIELD_DICT[short_year]
    s_inc, e_inc = fields["income"]
    s_bmi, e_bmi = fields["bmi"]

    try:
        values = {
            "education": getEducation(record[fields["education"] - 1]),
            "income": getIncome(record[s_inc - 1:e_inc]),
            "bmi": convertBMI(record[s_bmi - 1:e_bmi], short_year),
        }
        health = getHlth(record[fields["genhlth"] - 1])
    except (ValueError, IndexError, AssertionError):
        # garbled or truncated line, skip just this record
        return None

    # by default only the requested predictors need to be valid,
    # require_all=True copies the template exactly (all three must be valid)
    to_check = VALID_COLUMNS if require_all else columns
    if not is_valid("genhlth", health):
        return None
    if not all(is_valid(c, values[c]) for c in to_check):
        return None

    predictors = [float(values[c]) for c in columns]
    return predictors, float(health)


def iter_year_records(zip_path, short_year, columns, require_all=False):
    """yield clean observations from a single year's zip file."""
    columns = validate_columns(columns)

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        # each zip holds exactly one .ASC file (2014's name has a trailing space,
        # so we take it from namelist() instead of typing it out)
        assert len(members) == 1, f"expected one file inside {zip_path}, found {members}"

        # zf.open streams the compressed member, nothing is extracted to disk.
        # wrapping it in TextIOWrapper lets us loop over it line by line
        with zf.open(members[0]) as raw:
            for record in TextIOWrapper(raw, encoding="latin-1"):
                result = parse_record(record, short_year, columns, require_all)
                if result is not None:
                    yield result


def iter_health_records(data_dir, columns, require_all=False):
    """
    yield (predictors, health) for every valid record in 2011-2014.

    predictors is a list of floats in the same order as columns (no intercept).
    health is the genhlth value as a float (1 = excellent, 2 = very good, 3 = good,
           4 = fair, 5 = poor, and rarely 6).

    require_all controls which predictors must be valid for an observation to be included:
      - False (default): only the predictors in columns must be valid
                        (maximizes available data for each model)
      - True: all three predictors (education, income, bmi) must be valid
             (ensures consistent sample size across different model specifications,
              matches the behavior of the professor's linearRegression.py template)

    Example:
      If require_all=False and columns=['bmi'], an observation with valid bmi
      but missing income is INCLUDED.
      If require_all=True, the same observation is EXCLUDED.
    """
    columns = validate_columns(columns)
    paths = find_zip_files(data_dir)

    for year in YEARS:
        yield from iter_year_records(paths[year], year, columns, require_all)


def count_by_year(data_dir, columns, require_all=False):
    """count valid observations per year, handy for sanity checks."""
    columns = validate_columns(columns)
    paths = find_zip_files(data_dir)
    return {2000 + year: sum(1 for _ in iter_year_records(paths[year], year, columns, require_all))
            for year in YEARS}


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    cols = ["education", "income", "bmi"]

    records = iter_health_records(data_dir, cols)
    print("first 3 observations:")
    for _ in range(3):
        print(" ", next(records))

    counts = count_by_year(data_dir, cols)
    for year, n in counts.items():
        print(year, n)
    print("total", sum(counts.values()))
