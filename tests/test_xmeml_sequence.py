"""The xmeml reader takes the edit itself, not a nested sequence the file lists first (Premiere XtoCC export, 2611 LNGSHOT BEEP BEEP)."""
from contour.core.timeline import read_timeline

XML = """<?xml version="1.0" encoding="UTF-8"?>
<xmeml version="4">
 <sequence id="n1"><name>title nest</name><duration>20</duration><rate><timebase>24</timebase><ntsc>TRUE</ntsc></rate>
  <media><video><track><clipitem id="t1"><name>logo</name><start>0</start><end>20</end><in>0</in><out>20</out></clipitem></track></video></media>
 </sequence>
 <sequence id="main"><name>EDIT</name><duration>80</duration><rate><timebase>24</timebase><ntsc>TRUE</ntsc></rate>
  <media><video><track>
   <clipitem id="a"><name>A_0001</name><start>0</start><end>30</end><in>100</in><out>130</out><file id="f1"><name>A_0001.mxf</name></file></clipitem>
   <clipitem id="b"><name>A_0002</name><start>30</start><end>60</end><in>200</in><out>230</out><file id="f2"><name>A_0002.mxf</name></file></clipitem>
   <clipitem id="c"><name>title nest</name><start>60</start><end>80</end><in>0</in><out>20</out><sequence id="n1"/></clipitem>
  </track></video></media>
 </sequence>
</xmeml>
"""


def test_the_edit_not_the_nested_sequence(tmp_path):
    p = tmp_path / "edit.xml"
    p.write_text(XML, encoding="utf-8")
    tl = read_timeline(str(p), 23.976)
    assert tl.name == "EDIT"
    assert [(s.rec_in, s.rec_out) for s in tl.shots if s.kind == "clip"][:2] == [(0, 30), (30, 60)]
    assert tl.duration == 80
