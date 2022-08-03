import random
_r = random.randint(-1,127)
from tkinter import *
import tkinter
root = Tk()
root.geometry("500x400")
root.title("UFC WORDLE")
result = ""
pickst = ""
pickstn = ""
win = ""
counter = '3'

from flask import Flask

app = Flask(__name__)

@app.route("/")
def function():
 fullname = {'FRANCIS NGANNOU':[0,'FRANCIS NGANNOU',35,0,8,6.4,83,0], 'CIRYL GANE':[1,'CIRYL GANE',32,1,8,6.4,81,4],#
 'STIPE MIOCIC':[2,'STIPE MIOCIC',39,2,8,6.4,80,2], 'TAI TUIVASA':[3,'TAI TUIVASA',29,3,8,6.2,75,4],#
 'CURTIS BLAYEDS':[4,'CURTIS BLAYEDS',31,4,8,6.4,80,3], 'DERRICK LEWIS':[5,'DERRICK LEWIS',37,5,8,6.3,79,2],#
 'TOM ASPHINALL':[6,'TOM ASPHINALL',29,6,8,6.5,78,2], 'ALEXANDER VOLKOV':[7,'ALEXANDER VOLKOV',33,7,8,6.7,80,3],#
 'JAIRZINHO ROZENSTRUIK':[8,'JAIRZINHO ROZENSTRUIK',34,8,8,6.2,78,1], 'CHRIS DAUKAUS':[9,'CHRIS DAUKAUS',32,9,8,6.3,76,2],#
 'MARCIN TYBURA':[10,'MARCIN TYBURA',36,10,8,6.3,78,2], 'SERGEI PAVLOVICH':[11,'SERGEI PAVLOVICH',30,11,8,6.3,84,2,],#
 'ALEXANDER ROMANOV':[12,'ALEXANDER ROMANOV',31,12,8,6.2,75,2], 'SHAMIL ABDURAKHIMOV':[13,'SHAMIL ABDURAKHIMOV',40,13,8,6.3,76,1],#
 'AUGUSTO SAKAI':[14,'AUGUSTO SAKAI',31,14,8,6.4,77,2], 'BLAGOY IVANOV':[15,'BLAGOY IVANOV',35,15,8,5.9,73,2],#
 'JIRI PROCHAZKA':[16,'JIRI PROCHAZKA',29,0,7,6.3,80,4], 'GLOVER TEIXEIRA':[17,'GLOVER TEIXEIRA',42,1,7,6.2,76,2],#
 'JAN BLACHOWICZ':[18,'JAN BLACHOWICZ',39,2,7,6.2,78,4], 'ALEKSANDAR RAKIC':[19,'ALEKSANDAR RAKIC',30,3,7,6.5,78,2],#
 'MAGOMED ANKALAEV':[20,'MAGOMED ANKALAEV',30,4,7,6.3,75,3], 'ANTHONY SMITH':[21,'ANTHONY SMITH',33,5,7,6.4,76,3],#
 'THIAGO SANTOS':[22,'THIAGO SANTOS',38,6,7,6.2,76,3], 'DOMINICK REYES':[23,'DOMINICK REYES',32,7,7,6.4,77,3],#
 'PAUL CRAIG':[24,'PAUL CRAIG',34,8,7,6.4,76,2], 'VOLKAN OEZDEMIR':[25,'VOLKAN OEZDEMIR',32,9,7,6.1,75,1],#
 'JAMAHAL HILL':[26,'JAMAHAL HILL',31,10,7,6.4,79,2], 'NIKITA KRYLOV':[27,'NIKITA KRYLOV',30,11,7,6.3,77.5,3],#
 'RYAN SPANN':[28,'RYAN SPANN',30,12,7,6.5,79,2], 'JOHHNY WALKER':[29,'JOHHNY WALKER',30,13,7,6.6,82,2],#
 'JIMMY CRUTE':[30,'JIMMY CRUTE',26,14,7,6.2,74,2], 'DUSTIN JACOBY':[31,'DUSTIN JACOBY',34,15,7,6.4,76,2],#
 'ISRAEL ADESANYA':[32,'ISRAEL ADESANYA',32,0,6,6.4,80,2], 'ROBER WHITAKER':[33,'ROBER WHITAKER',31,1,6,6.0,73,3],#
 'JARED CANNONIER':[34,'JARED CANNONIER',38,2,6,6.0,77.5,3], 'MARVIN VETTORI':[35,'MARVIN VETTORI',28,3,6,6.0,74,2],#
 'DEREK BRUNSON':[36,'DEREK BRUNSON',38,4,6,6.1,77,3], 'PAULO COSTA':[37,'PAULO COSTA',31,5,6,6.0,72,2],#
 'ALEX PEREIRA':[38,'ALEX PEREIRA',35,6,6,6.4,79,3],'SEAN STRICKLAND':[39,'SEAN STRICKLAND',31,7,6,6.1,76,3],#
 'JACK HERMANSON':[40,'JACK HERMANSON',34,8,6,6.1,77,3], 'DARREN TILL':[41,'DARREN TILL',29,9,6,6.0,74,4],#
 'ANDRE MUNIZ':[42,'ANDRE MUNIZ',32,10,6,6.1,78,3], 'KEVIN GASTELUM':[43,'KEVIN GASTELUM',30,11,6,5.9,71.5,3],# 
 'URIAH HALL':[44,'URIAH HALL',37,12,6,6.0,79.5,1], 'NASSOURDINE IMAVOV':[45,'NASSOURDINE IMAVOV',27,13,6,6.3,75,0],#
 'DRICUS PLESSIS':[46,'DRICUS PLESSIS',28,14,6,6.1,76,3], 'BRAD TAVARES':[47,'BRAD TAVARES',34,15,6,5.911,74,3],#
 'KAMARU USMAN':[48,'KAMARU USMAN',35,0,5,6.0,76,2], 'COLBY COVINGTON':[49,'COLBY COVINGTON',34,1,5,5.911,72,2],#
 'LEON EDWARDS':[50,'LEON EDWARDS',30,2,5,6.0,74,1], 'KHAMZAT CHIMAEV':[51,'KHAMZAT CHIMAEV',28,3,5,6.2,75,2],#
 'GILBERT BURNS':[52,'GILBERT BURNS',36,4,5,5.910,71,3], 'BELAL MUHAMMAD':[53,'BELAL MUHAMMAD',34,5,5,5.910,72,3],# 
 'VICENTE LUQUE':[54,'VICENTE LUQUE',30,6,5,5.911,75.5,2], 'STEPHEN THOMPSON':[55,'STEPHEN THOMPSON',39,7,5,6.0,75,1],#
 'JORGE MASVIDAL':[56,'JORGE MASVIDAL',37,8,5,5.911,74,0], 'SEAN BRADY':[57,'SEAN BRADY',29,9,5,5.910,72,2],#
 'SHAVKAT RAKHMONOV':[58,'SHAVKAT RAKHMONOV',27,10,5,6.1,77,2], 'MICHAEL CHIESA':[59,'MICHAEL CHIESA',34,11,5,6.1,75.5,2],#
 'NEIL MAGNY':[60,'NEIL MAGNY',34,12,5,6.3,80,3], 'GEOFF NEAL':[61,'GEOFF NEAL',31,13,5,5.911,75,3],#
 'LI JINGLIANG':[62,'LI JINGLIANG',34,14,5,6.0,72,3], 'MICHEL PEREIRA':[63,'MICHEL PEREIRA',28,15,5,6.1,73,2],#
 'CHARLES OLIVEIRA':[64,'CHARLES OLIVEIRA',32,1,4,5.910,74,2], 'DUSTIN POIRIER':[65,'DUSTIN POIRIER',33,2,4,5.9,72,3],#
 'JUSTIN GAETHJE':[66,'JUSTIN GAETHJE',33,3,4,5.911,70,3], 'ISLAM MAKHACHEV':[67,'ISLAM MAKHACHEV',30,4,4,5.910,70,2],# 
 'MICHAEL CHANDLER':[68,'MICHAEL CHANDLER',36,5,4,5.8,71,2], 'BENEIL DARIUSH':[69,'BENEIL DARIUSH',33,6,4,5.910,72,3],#
 'RAFAEL FIZIEV':[70,'RAFAEL FIZIEV',29,7,4,5.8,71.5,4], 'RAFAEL DOS ANJOS':[71,'RAFAEL DOS ANJOS',37,8,4,5.8,70,3],#
 'MATEUSZ GAMROT':[72,'MATEUSZ GAMROT',31,9,4,5.910,71,2], 'ARMAN TSARUKYAN':[73,'ARMAN TSARUKYAN',25,10,4,5.7,72,3],#
 'TONY FERGUSON':[74,'TONY FERGUSON',38,11,4,5.91,76.5,2], 'CONOR MCGREGOR':[75,'CONOR MCGREGOR',34,12,4,5.8,74,1],#
 'DAN HOOKER':[76,'DAN HOOKER',32,13,4,6.0,75,2], 'JALIN TURNER':[77,'JALIN TURNER',27,14,4,6.3,77,3],#
 'DAMIR ISMAGULOV':[78,'DAMIR ISMAGULOV',31,15,4,5.910,74,4], 'ALEXANDER VOLKANOVSKI':[79,'ALEXANDER VOLKANOVSKI',33,0,3,5.6,71,4],#
 'MAX HOLLOWAY':[80,'MAX HOLLOWAY',30,1,3,5.911,69,3], 'YAIR RODRIGUEZ':[81,'YAIR RODRIGUEZ',29,2,3,5.911,71,1],#
 'BRIAN ORTEGA':[82,'BRIAN ORTEGA',31,3,3,5.8,69,2], 'JOSH EMMETT':[83,'JOSH EMMETT',37,4,3,5.6,70,2],#
 'CALVIN KATTAR':[84,'CALVIN KATTAR',34,5,3,5.911,72,0], 'ARNOLD ALLEN':[85,'ARNOLD ALLEN',28,6,3,5.8,70,3],#
 'KOREAN ZOMBIE':[86,'KOREAN ZOMBIE',35,7,3,5.7,72,3], 'GIGA CHIKADZE':[87,'GIGA CHIKADZE',33,8,3,6.0,74,1],#
 'BRYCE MITCHELL':[88,'BRYCE MITCHELL',27,9,3,5.9,70,2], 'MOVSAR EVLOEV':[89,'MOVSAR EVLOEV',28,10,3,5.8,72,2],#
 'DAN IGE':[90,'DAN IGE',30,11,3,5.7,71,2], 'SODIQ YUSUFF':[91,'SODIQ YUSUFF',29,12,3,5.9,71,3],#
 'EDSON BARBOZA':[92,'EDSON BARBOZA',36,13,3,5.911,75,2], 'SHANE BURGOS':[93,'SHANE BURGOS',31,14,3,5.911,75.5,3],#
 'ILIA TOPURIA':[94,'ILIA TOPURIA',25,15,3,5.7,69,2], 'ALJAMAIN STERLING':[95,'ALJAMAIN STERLING',32,0,2,5.7,71,2],#
 'PETR YAN':[96,'PETR YAN',29,1,2,5.7,67,4], 'TJ DILLASHAW':[97,'TJ DILLASHAW',36,2,2,5.6,67,2],#
 'JOSE ALDO':[98,'JOSÉ ALDO',35,3,2,5.7,70,2], 'CORY SANDHAGEN':[99,'CORY SANDHAGEN',30,4,2,5.911,70,3],#
 'MARLON VERA':[100,'MARLON VERA',29,5,2,5.8,70.5,2], 'MERAB DVALISHVILI':[101,'MERAB DVALISHVILI',31,6,2,5.6,68,2],#
 'ROB FONT':[102,'ROB FONT',35,7,2,5.8,71.5,2], 'DOMINICK CRUZ':[103,'DOMINICK CRUZ',37,8,2,5.8,68,0],#
 'PEDRO MUNHOZ':[104,'PEDRO MUNHOZ',35,9,2,5.6,65,2], 'SONG YADONG':[105,'SONG YADONG',24,10,2,5.8,67,1],#
 'RICKY SIMON':[106,'RICKY SIMON',29,11,2,5.6,2], 'FRANKIE EDGAR':[107,'FRANKIE EDGAR',40,12,2,5.6,68,2],#
 'SEAN OMALLEY':[108,'SEAN OMALLEY',27,13,2,5.911,72,3], 'UMAR NURMAGOMEDOV':[109,'UMAR NURMAGOMEDOV',26,14,2,5.8,69,2],#
 'JACK SHORE':[110,'JACK SHORE',27,15,2,5.9,71.0,2], 'DEIVESON FIGUEIRDO':[111,'DEIVESON FIGUEIRDO',34,0,1,5.5,68,3],#
 'BRANDON MORENO':[112,'BRANDON MORENO',28,1,1,5.7,70,2], 'KAI KARA FRANCE':[113,'KAI KARA',29,2,1,5.5,69,2],#
 'ASKAR ASKAROV':[114,'ASKAR ASKAROV',29,3,1,5.6,67,2], 'ALEXANDRE PANTOJA':[115,'ALEXANDRE PANTOJA',32,4,1,5.5,67,2],#
 'BRANDON ROYVAL':[116,'BRANDON ROYVAL',29,5,1,5.7,68,2], 'ALEX PEREZ':[117,'ALEX PEREZ',30,6,1,5.6,65,2],# 
 'MATHEUS NICOLAU':[118,'MATHEUS NICOLAU',29,7,1,5.6,66,2], 'MATT SCHNELL':[119,'MATT SCHNELL',32,8,1,5.8,70,2],#
 'DAVID DVORAK':[120,'DAVID DVORAK',30,9,1,5.5,68,3], 'TIM ELLIOTT':[121,'TIM ELLIOTT',35,10,1,5.7,66,2],#
 'AMIR ALBAZI':[122,'AMIR ALBAZI',28,11,1,5.6,68,3], 'SU MUDAERJI':[123,'SU MUDAERJI',26,12,1,5.8,72,1],#
 'MANEL KAPE':[124,'MANEL KAPE',28,13,1,5.5,68,3], 'JEFFREY MOLINA':[125,'JEFFREY MOLINA',25,14,1,5.6,69,3],#
 'TAGIR ULANBEKOV':[126,'TAGIR ULANBEKOV',31,15,1,5.7,70,2]
 } 


 number = {0:[0,'FRANCIS NGANNOU',35,0,8,6.4,83,0], 1:[1,'CIRYL GANE',32,1,8,6.4,81,4],#
 2:[2,'STIPE MIOCIC',39,2,8,6.4,80,2], 3:[3,'TAI TUIVASA',29,3,8,6.2,75,4],#
 4:[4,'CURTIS BLAYEDS',31,4,8,6.4,80,3], 5:[5,'DERRICK LEWIS',37,5,8,6.3,79,2],#
 6:[6,'TOM ASPHINALL',29,6,8,6.5,78,2], 7:[7,'ALEXANDER VOLKOV',33,7,8,6.7,80,3],#
 8:[8,'JAIRZINHO ROZENSTRUIK',34,8,8,6.2,78,1], 9:[9,'CHRIS DAUKAUS',32,9,8,6.3,76,2],#
 10:[10,'MARCIN TYBURA',36,10,8,6.3,78,2], 11:[11,'SERGEI PAVLOVICH',30,11,8,6.3,84,2,],#
 12:[12,'ALEXANDER ROMANOV',31,12,8,6.2,75,2], 13:[13,'SHAMIL ABDURAKHIMOV',40,13,8,6.3,76,1],#
 14:[14,'AUGUSTO SAKAI',31,14,8,6.4,77,2], 15:[15,'BLAGOY IVANOV',35,15,8,5.9,73,2],#
 16:[16,'JIRI PROCHAZKA',29,0,7,6.3,80,4], 17:[17,'GLOVER TEIXEIRA',42,1,7,6.2,76,2],#
 18:[18,'JAN BLACHOWICZ',39,2,7,6.2,78,4], 19:[19,'ALEKSANDAR RAKIC',30,3,7,6.5,78,2],#
 20:[20,'MAGOMED ANKALAEV',30,4,7,6.3,75,3], 21:[21,'ANTHONY SMITH',33,5,7,6.4,76,3],#
 22:[22,'THIAGO SANTOS',38,6,7,6.2,76,3], 23:[23,'DOMINICK REYES',32,7,7,6.4,77,3],#
 24:[24,'PAUL CRAIG',34,8,7,6.4,76,2], 25:[25,'VOLKAN OEZDEMIR',32,9,7,6.1,75,1],#
 26:[26,'JAMAHAL HILL',31,10,7,6.4,79,2], 27:[27,'NIKITA KRYLOV',30,11,7,6.3,77.5,3],#
 28:[28,'RYAN SPANN',30,12,7,6.5,79,2], 29:[29,'JOHHNY WALKER',30,13,7,6.6,82,2],#
 30:[30,'JIMMY CRUTE',26,14,7,6.2,74,2], 31:[31,'DUSTIN JACOBY',34,15,7,6.4,76,2],#
 32:[32,'ISRAEL ADESANYA',32,0,6,6.4,80,2], 33:[33,'ROBER WHITAKER',31,1,6,6.0,73,3],#
 34:[34,'JARED CANNONIER',38,2,6,6.0,77.5,3], 35:[35,'MARVIN VETTORI',28,3,6,6.0,74,2],#
 36:[36,'DEREK BRUNSON',38,4,6,6.1,77,3], 37:[37,'PAULO COSTA',31,5,6,6.0,72,2],#
 38:[38,'ALEX PEREIRA',35,6,6,6.4,79,3],39:[39,'SEAN STRICKLAND',31,7,6,6.1,76,3],#
 40:[40,'JACK HERMANSON',34,8,6,6.1,77,3], 41:[41,'DARREN TILL',29,9,6,6.0,74,4],#
 42:[42,'ANDRE MUNIZ',32,10,6,6.1,78,3], 43:[43,'KEVIN GASTELUM',30,11,6,5.9,71.5,3],# 
 44:[44,'URIAH HALL',37,12,6,6.0,79.5,1], 45:[45,'NASSOURDINE IMAVOV',27,13,6,6.3,75,0],#
 46:[46,'DRICUS PLESSIS',28,14,6,6.1,76,3], 47:[47,'BRAD TAVARES',34,15,6,5.911,74,3],#
 48:[48,'KAMARU USMAN',35,0,5,6.0,76,2], 49:[49,'COLBY COVINGTON',34,1,5,5.911,72,2],#
 50:[50,'LEON EDWARDS',30,2,5,6.0,74,1], 51:[51,'KHAMZAT CHIMAEV',28,3,5,6.2,75,2],#
 52:[52,'GILBERT BURNS',36,4,5,5.910,71,3], 53:[53,'BELAL MUHAMMAD',34,5,5,5.910,72,3],# 
 54:[54,'VICENTE LUQUE',30,6,5,5.911,75.5,2], 55:[55,'STEPHEN THOMPSON',39,7,5,6.0,75,1],#
 56:[56,'JORGE MASVIDAL',37,8,5,5.911,74,0], 57:[57,'SEAN BRADY',29,9,5,5.910,72,2],#
 58:[58,'SHAVKAT RAKHMONOV',27,10,5,6.1,77,2], 59:[59,'MICHAEL CHIESA',34,11,5,6.1,75.5,2],#
 60:[60,'NEIL MAGNY',34,12,5,6.3,80,3], 61:[61,'GEOFF NEAL',31,13,5,5.911,75,3],#
 62:[62,'LI JINGLIANG',34,14,5,6.0,72,3], 63:[63,'MICHEL PEREIRA',28,15,5,6.1,73,2],#
 64:[64,'CHARLES OLIVEIRA',32,1,4,5.910,74,2], 65:[65,'DUSTIN POIRIER',33,2,4,5.9,72,3],#
 66:[66,'JUSTIN GAETHJE',33,3,4,5.911,70,3], 67:[67,'ISLAM MAKHACHEV',30,4,4,5.910,70,2],# 
 68:[68,'MICHAEL CHANDLER',36,5,4,5.8,71,2], 69:[69,'BENEIL DARIUSH',33,6,4,5.910,72,3],#
 70:[70,'RAFAEL FIZIEV',29,7,4,5.8,71.5,4], 71:[71,'RAFAEL DOS ANJOS',37,8,4,5.8,70,3],#
 72:[72,'MATEUSZ GAMROT',31,9,4,5.910,71,2], 73:[73,'ARMAN TSARUKYAN',25,10,4,5.7,72,3],#
 74:[74,'TONY FERGUSON',38,11,4,5.91,76.5,2], 75:[75,'CONOR MCGREGOR',34,12,4,5.8,74,1],#
 76:[76,'DAN HOOKER',32,13,4,6.0,75,2], 77:[77,'JALIN TURNER',27,14,4,6.3,77,3],#
 78:[78,'DAMIR ISMAGULOV',31,15,4,5.910,74,4], 79:[79,'ALEXANDER VOLKANOVSKI',33,0,3,5.6,71,4],#
 80:[80,'MAX HOLLOWAY',30,1,3,5.911,69,3], 81:[81,'YAIR RODRIGUEZ',29,2,3,5.911,71,1],#
 82:[82,'BRIAN ORTEGA',31,3,3,5.8,69,2], 83:[83,'JOSH EMMETT',37,4,3,5.6,70,2],#
 84:[84,'CALVIN KATTAR',34,5,3,5.911,72,0], 85:[85,'ARNOLD ALLEN',28,6,3,5.8,70,3],#
 86:[86,'KOREAN ZOMBIE',35,7,3,5.7,72,3], 87:[87,'GIGA CHIKADZE',33,8,3,6.0,74,1],#
 88:[88,'BRYCE MITCHELL',27,9,3,5.9,70,2], 89:[89,'MOVSAR EVLOEV',28,10,3,5.8,72,2],#
 90:[90,'DAN IGE',30,11,3,5.7,71,2], 91:[91,'SODIQ YUSUFF',29,12,3,5.9,71,3],#
 92:[92,'EDSON BARBOZA',36,13,3,5.911,75,2], 93:[93,'SHANE BURGOS',31,14,3,5.911,75.5,3],#
 94:[94,'ILIA TOPURIA',25,15,3,5.7,69,2], 95:[95,'ALJAMAIN STERLING',32,0,2,5.7,71,2],#
 96:[96,'PETR YAN',29,1,2,5.7,67,4], 97:[97,'TJ DILLASHAW',36,2,2,5.6,67,2],#
 98:[98,'JOSÉ ALDO',35,3,2,5.7,70,2], 99:[99,'CORY SANDHAGEN',30,4,2,5.911,70,3],#
 100:[100,'MARLON VERA',29,5,2,5.8,70.5,2], 101:[101,'MERAB DVALISHVILI',31,6,2,5.6,68,2],#
 102:[102,'ROB FONT',35,7,2,5.8,71.5,2], 103:[103,'DOMINICK CRUZ',37,8,2,5.8,68,0],#
 104:[104,'PEDRO MUNHOZ',35,9,2,5.6,65,2], 105:[105,'SONG YADONG',24,10,2,5.8,67,1],#
 106:[106,'RICKY SIMON',29,11,2,5.6,2], 107:[107,'FRANKIE EDGAR',40,12,2,5.6,68,2],#
 108:[108,'SEAN OMALLEY',27,13,2,5.911,72,3], 109:[109,'UMAR NURMAGOMEDOV',26,14,2,5.8,69,2],#
 110:[110,'JACK SHORE',27,15,2,5.9,71.0,2], 111:[111,'DEIVESON FIGUEIRDO',34,0,1,5.5,68,3],#
 112:[112,'BRANDON MORENO',28,1,1,5.7,70,2], 113:[113,'KAI KARA',29,2,1,5.5,69,2],#
 114:[114,'ASKAR ASKAROV',29,3,1,5.6,67,2], 115:[115,'ALEXANDRE PANTOJA',32,4,1,5.5,67,2],#
 116:[116,'BRANDON ROYVAL',29,5,1,5.7,68,2], 117:[117,'ALEX PEREZ',30,6,1,5.6,65,2],# 
 118:[118,'MATHEUS NICOLAU',29,7,1,5.6,66,2], 119:[119,'MATT SCHNELL',32,8,1,5.8,70,2],#
 120:[120,'DAVID DVORAK',30,9,1,5.5,68,3], 121:[121,'TIM ELLIOTT',35,10,1,5.7,66,2],#
 122:[122,'AMIR ALBAZI',28,11,1,5.6,68,3], 123:[123,'SU MUDAERJI',26,12,1,5.8,72,1],#
 124:[124,'MANEL KAPE',28,13,1,5.5,68,3], 125:[125,'JEFFREY MOLINA',25,14,1,5.6,69,3],#
 126:[126,'TAGIR ULANBEKOV',31,15,1,5.7,70,2] }


 global result
 global pickst
 global counter
 userpic = entry1.get()
 userpic = userpic.upper()
 try:
  fullname[userpic]
 except KeyError as ex:
  ex = "Not a valid fighter!"
  label5 = Label(root, text = ex, bg="red").pack()
 userpick = fullname[userpic]

 answer = number[_r]
 age = userpick[2]
 rank = userpick[3]
 div = userpick[4]
 height = userpick[5]
 reach = userpick[6]
 style = userpick[7]
 agep = answer[2]
 rankp = answer[3]
 divp = answer[4]
 heightp = answer[5]
 reachp = answer[6]
 stylep = answer[7]

 
 pickst += str(userpick[1])
 pickst += "\t"
 pickst += str(userpick[2])
 pickst += "\t"
 pickst += str(userpick[3])
 pickst += "\t"
 if userpick[4] == 8:
  pickst += "Heavyweight"
 elif userpick[4] == 7:
  pickst += "Light Heavyweight"
 elif userpick[4] == 6:
  pickst += "Middleweight"
 elif userpick[4] == 5:
  pickst += "Welterweight"
 elif userpick[4] == 4:
  pickst += "Lightweight"
 elif userpick[4] == 3:
  pickst += "Featherweight"
 elif userpick[4] == 2:
  pickst += "Bantamweight"
 elif userpick[4] == 1:
  pickst += "flyweight"
 pickst += "\t"
 if userpick[5] == 5.910:
  pickst += "5'10"
 elif userpick[5] == 5.911:
  pickst += "5'11"
 else:
  pickst += str(userpick[5])
 pickst += "\t"
 pickst += str(userpick[6])
 pickst += "\t"
 if userpick[7] == 0:
  pickst += "Boxer"
 elif userpick[7] == 1:
  pickst += "Kickoxer"
 elif userpick[7] == 2:
  pickst += "Grapple"
 elif userpick[7] == 3:
  pickst += "MMA"
 elif userpick[7] == 4:
  pickst += "Muy Thai"

 
 if age < agep:
  result += "\t\t"
  result += u'\u2191'
 elif agep < age: 
  result += "\t\t"
  result += u'\u2193'
 else:
  result += "\t\t"
  result += "="
 if rank < rankp:
  result += "\t" 
  result += u'\u2193'
 elif rankp < rank:
  result += "\t"
  result += u'\u2191'
 else:
  result += "\t"
  result += "="
 if div < divp: 
  result += "\t"
  result += u'\u2191'
 elif divp < div:
  result += "\t"
  result += u'\u2193'
 else:
  result += "\t"
  result += "=" 
 if height < heightp:
  result += "\t\t"
  result += u'\u2191'
 elif heightp < height: 
  result += "\t\t"
  result += u'\u2193'
 else: 
  result += "\t\t"
  result += "=" 
 if reach < reachp:
  result += "\t"
  result += u'\u2191'
 elif reachp < reach: 
  result += "\t"
  result += u'\u2193'
 else:
  result += "\t"
  result += "=" 
 if style < stylep:
  result += "\t"
  result += "no"
  result += "\n"
 elif stylep < style:
  result += "\t"
  result += "no"
  result += "\n"
 else:
  result += "\t"
  result += "="
  result += "\n"
 
 counter = int(counter)
 if counter > 0:
  label3 = Label(root, text=result + "\n").pack(anchor='w')
  label4 = Label(root, text=pickst + "\n").pack(anchor='w')
  del result
  result = ""
  del pickst
  pickst = ""
  counter = counter-1
  counter = str(counter)
 else:
  answer = str(answer)
  label3 = Label(root, text="Sorry no attempts left\nAnswer: "+answer).pack() 
 

 if answer == userpick:
  win = "\nCongratulations: " + str(counter) + " attempts left"
  label5 = Label(root, text=win, bg="green")

label1 = Label(root, text="Please enter (Firstname space Lastname):").pack
entry1 = Entry(root)
entry1.pack()
button2 = Button(root, text="confirm", command=function).pack()
label2 = Label(root, text="Name\t\tage\trank\tdivision\t\theight\treach\tstyle").pack(anchor='w')




root.mainloop() 


