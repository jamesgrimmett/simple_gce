import pytest

from simple_gce.utils import chem_elements


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("Cr55", ("Cr", 55)),
        ("4He", ("He", 4)),
        ("^12^C", ("C", 12)),
        ("C^13", ("C", 13)),
        ("C(13)", ("C", 13)),
        ("^56^Co", ("Co", 56)),
        ("Co^56", ("Co", 56)),
        ("Co", ("Co", None)),
        ("H", ("H", None)),
    ],
)
def test_parse_chemical_symbol(symbol, expected):
    el, mass_num = chem_elements.parse_chemical_symbol(symbol)
    assert el, mass_num == expected


@pytest.mark.parametrize("symbol", ["1", "12Carbon", "Hydrogen"])
def test_parse_chemical_symbol_silent(symbol):
    result = chem_elements.parse_chemical_symbol(symbol, silent=True)
    assert result == (None, None)


@pytest.mark.parametrize("symbol", ["1", "12Carbon", "Hydrogen"])
def test_parse_chemical_symbol_error(symbol):
    with pytest.raises(ValueError):
        chem_elements.parse_chemical_symbol(symbol)
