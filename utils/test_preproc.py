from ._name_pre import *

def test_preproc(): 
    assert name_preprocessing("CANON KABUSHIKI KAISHA")=="canon matchit inc"
    assert name_preprocessing("SIEMENS AKTIENGESELLSCHAFT")=="siemens matchit inc"
    assert name_preprocessing('MATSUSHITA ELECTRIC INDUSTRIAL CO., LTD.')=='matsushita electric industries inc'
    assert name_preprocessing('KABUSHIKI KAISHA TOSHIBA')=='toshiba matchit'
    assert name_preprocessing('GENERAL ELECTRIC COMPANY')=='general electric inc'
    assert name_preprocessing('EASTMAN KODAK COMPANY')=='kodak matchit inc'
    assert name_preprocessing('MITSUBISHI DENKI KABUSHIKI KAISHA')=='mitsubishi denki inc'
    assert name_preprocessing('FUJITSU LIMITED')=='fujitsu inc'
    assert name_preprocessing('ROBERT BOSCH GMBH')=='robert bosch inc'
    assert name_preprocessing('BASF AKTIENGESELLSCHAFT')=='basf inc'
    assert name_preprocessing('KONINKLIJKE PHILIPS ELECTRONICS N.V.')=='koninklijke philips electronics inc'
    assert name_preprocessing('SAMSUNG ELECTRONICS CO., LTD.')=='samsung electronics inc'
    assert name_preprocessing('FUJI PHOTO FILM CO., LTD.')=='fuji photo film inc'
    assert name_preprocessing('HEWLETT-PACKARD COMPANY')=='hewlet packard inc'
    assert name_preprocessing('BAYER AG')=='bayers inc'
    assert name_preprocessing('U.S. PHILIPS CORPORATION')=='usa philips inc'
    assert name_preprocessing('E.I. DU PONT DE NEMOURS AND COMPANY')=='du pont inc'
    assert name_preprocessing('PHILIPS ELECTRONICS N.V.')=='philips electronics inc'
    assert name_preprocessing('THE DOW CHEMICAL COMPANY')=='dow chemical inc'
    assert name_preprocessing('BAYER AKTIENGESELLSCHAFT')=='bayers inc'
    assert name_preprocessing('RICOH COMPANY, LTD.')=='ricoh inc'
    assert name_preprocessing('some s.a')=='some sa'
    assert name_preprocessing('some s a')=='some sa'
