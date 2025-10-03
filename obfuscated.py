import unittest
import math

class _Olo0IOlI_OIIlo0lo:

    def __init__(self, _Il10lIIo_oIolIl1O_oIl1llll_IOIO):
        self._Il10lIIo_oIolIl1O_oIl1llll_IOIO = _Il10lIIo_oIolIl1O_oIl1llll_IOIO

    def _1oIIII1o_Oo0IOoOo(self):
        return math.pi * self._Il10lIIo_oIolIl1O_oIl1llll_IOIO ** 2

    def _lO101OO1_oOOoOo1l_OlIlIo(self):
        return 4 * math.pi * self._Il10lIIo_oIolIl1O_oIl1llll_IOIO ** 2

    def _0IO0IIol_IIooI0o0_o1OOO0(self, _O1OlOIOo_0lolOo10):
        return 2 * math.pi * self._Il10lIIo_oIolIl1O_oIl1llll_IOIO * (self._Il10lIIo_oIolIl1O_oIl1llll_IOIO + _O1OlOIOo_0lolOo10)

    def _loOll101_1Ol1I0oO_OIIOol(self, _10lO1OO0_10IoooII):
        return self._Il10lIIo_oIolIl1O_oIl1llll_IOIO ** 2 * _10lO1OO0_10IoooII / 2

    def _0O1oIlI1_o0oolO01_oOl0IIOO_Ol0O(self, _olIloIlo_olo0l0Ol, _OlololOO_ll0o100I):
        return math.pi * (_OlololOO_ll0o100I ** 2 - _olIloIlo_olo0l0Ol ** 2)

class _IOolOIl1_o01loIOO_OO0lIOll_ooIl(unittest.TestCase):

    def _OIOoIoO0_llI0I1l1(self):
        _OOoIIllO_0I10oII0 = _Olo0IOlI_OIIlo0lo(0)
        return _OOoIIllO_0I10oII0._0O1oIlI1_o0oolO01_oOl0IIOO_Ol0O(1, 3)