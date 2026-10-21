import json

d = json.load(open(r'C:\Quant\atlas\.playwright-mcp\pk35_jaro.json', encoding='utf-8'))['data']
ms = d['markets']

want = ['854','855','856','857','858','859','863','864','865','450001','14','16','30','31','32',
        '15','25','21','551','548','549','550','18','12','13','11','10','29','1','45','41','26',
        '60000','900041','184','37','35','36','547','546','60300','60303','450004','23','24','8','9']
for m in ms:
    if m['id'] in want:
        outs = [(o['desc'], o['odds'], o.get('isActive', 1)) for o in (m.get('outcomes') or [])]
        print('=' * 100)
        print(f"id={m['id']} name={m.get('name')!r} desc={m.get('desc')!r} title={m.get('title')!r} group={m.get('group')!r} status={m.get('status')}")
        print(f"   guide={m.get('marketGuide')!r}")
        print(f"   outcomes={outs}")
