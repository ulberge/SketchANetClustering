(function() {

const b2 = [0.00997003,0.01395539,-0.02646736,0.04501361,-0.0032206,-0.00144896,-0.00121447,-0.0016972,0.00962637,0.00142821,-0.02143425,-0.00883297,-0.02006754,0.02115507,0.01262146,-0.00643008,0.01319264,0.00788097,0.00536033,-0.01116284,0.00520566,0.02433615,0.0203865,0.004645,0.10955355,0.03285444,0.01226468,-0.00436329,-0.02875906,-0.0113309,0.0204311,-0.01253099,-0.03247904,0.01159367,-0.0059413,0.01842359,-0.00948832,0.00747762,-0.02604991,0.00148086,-0.02206031,0.00351547,-0.0269109,-0.01142011,0.01323583,0.01427299,0.00930025,0.02024131,-0.0031393,0.0033983,0.01265,-0.01291717,0.02131166,0.00186469,-0.02288084,-0.0118063,-0.02254248,-0.01819161,-0.01997561,-0.02194779,0.00986785,0.01131453,-0.02519854,-0.01219932,0.00720305,-0.00055063,-0.01905347,0.01104095,0.00049273,0.00834797,0.01918209,0.02225368,-0.00581385,0.00638984,0.00812864,-0.05011284,0.00817468,-0.00663315,0.00953919,-0.01423915,-0.00240717,0.04653392,0.00088389,-0.0186742,-0.02212683,-0.00746176,0.00988964,0.03704196,0.01115131,0.00014331,0.00630318,-0.00641834,0.00577902,-0.00962109,0.00953234,-0.01856132,-0.00966473,-0.02253823,0.01120144,0.01651052,0.01182951,-0.05521365,0.01561194,0.00932868,-0.02350007,-0.05228635,0.01786828,0.001384,0.01123815,-0.01855011,0.01729783,-0.02053494,0.01365533,0.02272512,0.01849581,0.00627209,-0.01442522,0.00898677,-0.03129349,0.01925319,0.037666,-0.01156374,0.02382951,0.03353095,-0.00898111,-0.03185866,-0.01198433,-0.02167079];

  const b3 = [0.00841099303,0.000665222935,-0.0000806398821,0.00277150772,0.00919965841,0.0147868544,0.00653372798,-0.0109931231,0.0130885011,0.00351426541,0.0121856648,0.0186618045,0.0000925783024,-0.00599492248,0.00748247281,0.0197490677,-0.0053942604,0.00117850327,-0.000384311628,0.00215717009,0.00286614336,0.0125783775,0.0102907075,0.00113994058,0.00653604325,0.0223705024,0.001792836,0.00541230524,0.0031001505,0.00319501502,0.0152943786,0.000346581539,0.0103633795,0.00759810675,-0.0030007211,0.0213613752,0.012310029,0.0000903684995,0.00446101977,0.0111164646,0.0210042,0.015301236,0.00639730785,0.0212808549,-0.000760433555,0.00604506582,0.00191496534,-0.00196925248,0.0434228368,0.0117254918,0.0307395328,0.00937769562,0.00170705793,0.0175086297,0.0041514067,-0.00334055023,0.00229355809,0.0098363487,0.00865833182,0.0264375638,0.0192888007,0.00372490194,0.000884394336,0.00201801909,0.00599422911,0.0198435951,0.0051417416,0.0176345073,0.007937558,0.0381945297,-0.000115987641,0.0176899619,-0.0117437514,0.0421373583,0.00734366383,0.0522506051,-0.000876663718,0.020612143,0.00695610093,-0.00827727001,0.00910752639,0.00729219243,0.0238400474,0.0080795493,0.020599436,0.0303023905,0.0219575595,0.0131414328,0.0179013293,0.00999897439,0.0148735605,0.00369155779,0.00536017725,0.0131368926,0.000571071461,0.017816199,0.0135330306,-0.00775002781,0.0228619184,0.0329037271,0.014875439,-0.00923059229,0.00821943581,0.00193521637,0.00464618532,-0.00192261534,0.0084291026,0.000825655414,0.0492852069,0.00353882043,0.0297964681,0.00554571301,0.0114260018,0.00212390302,0.0222336259,0.0107367588,0.0453268625,0.00297140563,0.00938024744,0.00488852849,-0.000487849058,0.0189576223,0.00409993203,-0.00165351306,0.00339857163,0.00744869513,0.0185079966,0.019738568,0.039637778,-0.0046128449,0.0170144401,0.00518232724,-0.00223576301,0.00303568924,0.0119562577,0.00983450934,0.0102069713,0.00419994025,0.0127855018,0.0551044606,0.0112773599,0.0241218265,0.012275774,0.043691162,-0.00363699556,-0.00595172169,0.00387693383,0.0363456979,0.0437074155,0.00118624291,0.0076510217,-0.00479912898,0.0117759788,-0.00630635768,-0.00449216552,0.0106371213,0.00594309811,0.0112630865,0.00256862957,0.00487472769,0.00246499735,0.00157880096,0.0428670309,0.0141692581,0.0101267574,0.00150846166,0.000751364278,0.00435894821,0.011108879,0.0323591009,0.00725935632,0.00399368629,0.029856056,0.00592835527,0.00749743031,0.000770379382,0.0409357809,0.00769685395,0.00132092787,0.0199799575,0.0344173461,0.020319622,0.0340389535,0.00956068747,0.00894684438,0.0110219633,0.0230674837,-0.00125052175,0.00390948029,0.0199927744,0.0394904241,0.00509759597,0.00880051963,0.0054002339,-0.0021102326,0.00562179787,0.0406535976,0.0110611608,0.0196511671,0.0133259753,-0.00664195605,-0.00472992985,0.00939749088,0.00316081732,0.0263115596,0.0288194045,-0.00134996069,0.00244278926,0.0317835435,-0.00540148467,0.019048417,-0.00857265014,0.00575727783,0.00169536099,0.00301223784,0.0170070045,0.0293693412,0.0179603025,0.0100325719,0.0169640351,0.0391930938,-0.000834370439,0.0415241756,0.010066255,-0.00584430434,0.0086898813,0.00357487472,0.0295298174,0.000929674366,0.000951543974,0.00433411077,-0.00000275063758,0.00481357705,0.0143844904,0.00557127688,-0.00433353521,0.0487241372,0.0108484281,0.0242896192,-0.00151191955,0.00091326551,0.015841933,0.00656256359,0.0252370927,0.01844454,0.00190393836,0.00845259055,0.00312812207,0.0167634767,0.00566133531,0.0132076722,0.0104319453,0.0423489809,0.00305844937,0.0014817036,0.0220386926];

  const b4 = [0.00193618902,0.0151702752,0.0504706912,0.00635561999,0.00682744244,0.0327820331,0.0163175892,0.0180483777,0.0536312014,0.0146354456,0.0267920755,0.0259070825,0.0228385758,0.0189621039,0.0211810302,0.0241330974,0.0148323551,0.0247181989,-0.0139143858,0.0184182245,0.0508088097,0.0209409874,0.0312479511,0.0109915948,-0.0141369775,0.0550701469,0.0507929437,0.0265082121,0.00938558392,0.0479298718,0.0100476947,0.0133309802,0.0260826889,0.0152225671,0.0321506485,0.0179475956,0.01116772,0.0574212596,-0.0137288691,0.0275790952,0.0281763785,0.0210649148,0.0231124517,0.0395222418,-0.00988826994,0.0760268643,0.0560903549,0.0346367806,0.0218199734,0.0029436301,0.00438905321,0.0590581708,0.0131537113,0.0335520096,0.0478558019,0.0149502121,0.0366960913,0.0326431245,0.0863013119,0.0201825164,0.015933482,0.0292240959,0.0757032037,0.0284156445,0.0181196146,0.0357837118,0.0424486026,0.0290139485,-0.0118648149,0.0553282388,0.0639672875,0.0449442007,0.0363950506,0.0115426825,0.0191297084,-0.0000132945797,0.09263888,0.00941213872,-0.0102923708,0.0566890649,0.0323199891,0.0120134363,0.016573865,0.00399979623,0.0262924097,0.0942156017,-0.0195439011,0.0462970547,0.029496409,0.0165922977,-0.00488467002,0.000979031785,0.0358840488,0.0343966298,0.0123864934,0.00636532716,0.0161832571,0.026366448,0.00518400501,-0.0363691077,0.0163380355,0.0123144621,0.0173498038,-0.0027108572,0.0308598503,0.0670131445,-0.0162670389,0.0561857261,0.0222248174,0.117257647,0.00705969706,-0.00302915717,0.0437592939,0.0238568,0.0302582216,0.0117166191,0.0343298316,0.0290664528,0.0146238087,0.0111340415,0.0547069684,0.0193386842,0.0210748818,-0.0550246239,0.0118282884,0.0459401198,0.0453820266,0.0386801325,0.0205331724,0.0119571695,0.0217225943,0.106904805,0.00489290897,0.0053505199,0.00879999064,-0.00260655698,0.0406205915,0.0315686241,-0.00435713446,0.0421858579,0.0130195124,0.00096041162,0.0383816883,-0.0110490276,0.0669761375,-0.0109922625,-0.020386504,0.101245262,0.0410175659,-0.0026721789,0.0182386357,0.0442726314,0.0328008346,0.0256640166,0.051371824,0.0330180041,0.0165423118,0.0287954155,0.0221995041,0.0516734309,-0.00193431589,0.0387767442,0.0289229434,0.00750852935,0.0116059473,0.0287491642,0.0280618183,0.0590349287,0.0382127576,-0.0083785411,0.0179635994,0.0280611906,0.00839699991,0.0305287857,0.0181508213,0.0647372156,0.0234249234,-0.00933665782,-0.0318663046,-0.0313299038,-0.00788911991,0.063793987,0.0251532029,0.0360479876,-0.00291035511,0.0631615743,0.0283006076,0.0517580174,0.0199458599,0.0354801416,0.0484097488,-0.00879706349,-0.00780609436,0.0355038121,0.0282024313,0.0423637107,0.019488899,0.107875057,0.0423897132,0.0691873208,0.0129591385,0.0173777509,0.0676900819,0.0101919761,0.0176615547,0.0666321293,-0.024362687,0.0237208996,0.00640155608,0.0749717355,0.00726822019,0.0176120717,0.0251169521,0.0153386192,0.0327078812,0.0253830831,0.00211065356,0.0189761575,-0.0145663982,0.022554148,0.0173553899,-0.0065571419,0.0141206132,-0.0103561925,0.0578839369,0.0513491146,0.0359464288,0.0264963899,0.0100042252,0.0347510912,0.0407782644,0.0301854014,0.0301064719,0.0151727572,0.065223977,0.0830652639,-0.004069997,0.0357721895,-0.008560583,0.0267124996,0.013090116,0.0363673791,0.0305812974,0.0117289796,0.0156175364,0.00644924818,0.0172888953,0.0347458683,0.0302816965,0.00889527425,-0.0170252956,0.0322954096,0.0314142108,0.0299987961,0.0537286475,-0.00762776891];

  const b5 = [0.133780763,0.000815418491,0.0523970351,0.0451765545,0.0246841814,0.0401164442,0.0777170584,0.0171347652,-0.0540686846,0.0705855563,0.0424341522,0.0256427061,0.0665874109,0.128735811,0.0322389901,0.011533753,0.0825018063,0.00355670322,0.0369755737,0.0486872196,0.00672368705,0.119063348,0.0626749918,0.147441655,-0.0283411704,0.0264038071,0.0361780189,0.123828769,-0.0069676484,0.0275263563,0.029178502,0.068074964,-0.00907475688,0.00921118911,0.0257072374,0.0322153866,0.0269672964,0.100815721,0.00074435171,0.055817835,-0.0156131322,0.0200673845,-0.0245319083,0.0819070861,0.00180747791,0.0226963535,0.06842421,-0.00124100829,0.0172442403,0.0243951678,-0.0112641519,0.0300413407,-0.0125201363,0.0432866886,0.0511006005,-0.0222985912,-0.00185170455,0.0380515084,0.0332549736,0.0116565693,0.00096241961,0.00313232397,0.0503102019,0.0256751571,-0.00289120642,0.00498371618,0.0218040142,0.0353004523,0.0533175766,-0.0248596426,-0.00580624631,0.00634459965,0.0605166107,0.00436290912,0.0353538841,0.0134410514,0.027644597,0.0174017344,0.0305807032,0.071073398,0.0421986431,0.0160103031,0.0164016746,0.0403566658,0.0260267779,0.0127956588,0.0270114131,-0.00976312626,0.0990416408,0.00274051283,0.00775638176,0.0593378916,-0.0161060207,0.0240119137,0.0940764099,0.0730306879,0.0134814028,-0.00341071934,0.0100816153,-0.00401507178,0.0824829191,0.00686942274,-0.0132615613,-0.0276714787,0.10998974,-0.00737684872,0.0419307537,0.000201289382,0.0611421131,-0.00300904713,0.0466977954,0.00749033084,0.131491184,0.0087106796,0.122489437,0.0567007512,0.0623401143,0.0156676248,0.166523084,0.0791297555,0.000372980605,0.031391833,0.078472808,0.0647058487,-0.0474423617,0.239680916,-0.0144461282,-0.0149858845,0.0257089231,0.0423068479,-0.00580325117,0.017503636,-0.00251253322,0.0204469915,0.148912951,0.039325133,0.0173309967,0.0180385765,0.096938841,0.280715466,0.128043219,0.0111015299,0.0197066199,0.0277857073,0.0156668201,0.0589656606,0.0243905801,0.022550758,0.101141632,0.0120667247,0.0549639948,0.0336728059,0.00544091547,0.109467097,0.0218547843,0.0521973595,0.0390628763,0.0341493338,0.0327566117,0.0711642429,0.0741240233,0.0130575029,0.0470996685,0.0112365605,-0.00163571723,0.0213496499,0.162873372,0.138144091,0.0202902351,0.031062996,0.179978281,0.0122860922,0.0310544595,0.0130496565,0.00139729027,-0.00276091252,0.0347904935,0.00908301305,-0.0276058707,0.028606737,0.0335501991,0.0553773008,0.106495865,0.068128489,0.000916901685,0.079723075,0.0233757123,0.0621618368,0.0271413568,0.0803101659,0.0280197393,0.0113477036,0.0120990267,0.0486611389,0.00541203236,0.0549601354,0.103227198,0.0362395458,0.00377481221,-0.0452929102,0.0506165251,-0.000564605463,0.0609003156,0.0213688891,0.112996012,0.0312549248,0.0228162501,0.0319250934,0.0456766896,0.0117688579,0.0407019667,0.00205230806,0.0213182885,0.00210962282,0.00116458896,0.0798404068,0.0353475213,0.0975937843,0.0126710311,-0.000878624793,-0.00506915478,0.0365667753,-0.010629083,0.0499262139,0.0204116646,-0.00895794202,0.041104652,0.0463414527,0.156045794,0.00787041523,0.0700081587,0.0342480801,0.00950221438,-0.00784542877,0.0212634597,-0.00315694464,0.043171823,0.0361634605,-0.0273662917,0.24116087,0.0959478319,-0.0106693832,0.0200766418,0.135570765,-0.0796539858,0.0091443276,0.0923196152,0.0571016632,0.0258052628,0.147497639,0.023133453,-0.0399152897,0.132738367,0.00905458629,0.00599905523,0.110449694];

  // const data2 = b2.map((v, i) => {
  //   return { v: [v], i };
  // });
  // drawActivationGraph($('#bias2')[0], data2);
  // const data3 = b3.map((v, i) => {
  //   return { v: [v], i };
  // });
  // drawActivationGraph($('#bias3')[0], data3);
  // const d4 = b4.map((v, i) => {
  //   return { v: [v], i };
  // });
  // drawActivationGraph($('#bias4')[0], d4);
  // const d5 = b5.map((v, i) => {
  //   return { v: [v], i };
  // });
  // drawActivationGraph($('#bias5')[0], d5);

  // const biases = { b4, b5 };

  let currentMatchIndex = [0, 0, 0]; // layer and top match index

  document.onkeydown = e => {
    const maxTopMatch = $('.topMatches').eq(currentMatchIndex[0]).find('button').length;
    const maxConcept = $('.concepts').eq(currentMatchIndex[0]).find('button').length;
    if (e.key === 'a') {
      // a
      if (currentMatchIndex[2] > 0) {
        currentMatchIndex[2] -= 1;
        $('.topMatches').eq(currentMatchIndex[0]).find('button').eq(currentMatchIndex[2]).click();
      }
    } else if (e.key === 'd') {
      // d
      if (currentMatchIndex[2] < maxTopMatch - 1) {
        currentMatchIndex[2] += 1;
        $('.topMatches').eq(currentMatchIndex[0]).find('button').eq(currentMatchIndex[2]).click();
      }
    }

    if (e.key === 'w') {
        // w
      if (currentMatchIndex[1] > 0) {
        currentMatchIndex[1] -= 1;
        $('.concepts').eq(currentMatchIndex[0]).find('button').eq(currentMatchIndex[1]).click();
      }
    } else if (e.key === 's') {
        // s
      if (currentMatchIndex[1] < maxConcept - 1) {
        currentMatchIndex[1] += 1;
        $('.concepts').eq(currentMatchIndex[0]).find('button').eq(currentMatchIndex[1]).click();
      }
    }
    console.log(currentMatchIndex);
  };

  function drawActivationGraph(el, data) {
    const colors = ['#296a73', '#732642'];
    const color = '#000';
    const margin = {top: 10, right: 10, bottom: 10, left: 10};
    const width = Math.min(data.length * 100, el.offsetWidth - margin.left - margin.right) + margin.left + margin.right;
    const height = width * 0.35;

    const x = d3.scaleBand()
      .domain(data.map(d => d.i))
      .range([margin.left, width - margin.right])
      .padding(0.1);

    const y = d3.scaleLinear()
      .domain(d3.extent(data, d => d.v.length > 1 ? d.v[1] * 1.4 : d.v[0])).nice()
      .range([height - margin.bottom, margin.top]);

    const xAxis = g => g
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).tickSizeOuter(0));

    const svg = d3.select(el).append('svg')
      .attr('width', width)
      .attr('height', height);

    const bars = svg.append('g').selectAll('.bar').data(data).enter();

    const barCount = data[0].v.length;

    let minY = Infinity;
    data.forEach(d => d.v.forEach(v => v < minY ? minY = v : 0));

    for (let i = 0; i < barCount; i++) {
      bars.append('rect')
        .attr('fill', colors[i])
        .attr('x', d => x(d.i) + (i * (x.bandwidth() / barCount)))
        .attr('y', d => d.v[i] < 0 ? y(0) : y(d.v[i]))
        .attr('height', d => d.v[i] < 0 ? y(d.v[i]) - y(0) : y(0) - y(d.v[i]))
        .attr('width', x.bandwidth() / barCount);
    }

    const xAxisItems = svg.append('g')
          .style('font-size', '8px')
          .call(xAxis);

    xAxisItems.selectAll('.tick').remove();

    return svg.node();
  }

  async function getCentersData(fileName) {
    return new Promise((resolve) => {
      $.get(fileName, function(data) {
        const centersData = data.split('\n').map(acts => acts.split(',').map(v => v < 0 ? 0 : v));
        centersData.pop();
        resolve(centersData);
      });
    });
  }

  async function getTopMatchesData(fileName) {
    return new Promise((resolve) => {
      $.get(fileName, function(data) {
        const topMatchesData = data.split(':').map(g => g.trim().split('\n').map(acts => acts.split(',').map(v => v < 0 ? 0 : v)));
        topMatchesData.pop();
        // console.log(topMatchesData);
        resolve(topMatchesData);
      });
    });
  }

  function getConceptIcon(layerIndex, i) {
    if (layerIndex === 1) {
      const size = 35;
      const padding = 35;
      const rowSize = 16;
      const x = -padding - ((i % rowSize) * (size + padding));
      const y = -padding - (Math.floor(i / rowSize)* (size + padding));
      const conceptIconStyle = 'background-position: ' + x + 'px ' + y + 'px;';
      return conceptIconStyle;
    }
    if (layerIndex === 2) {
      const size = 35;
      const padding = 35;
      const rowSize = 16;
      const x = -padding - ((i % rowSize) * (size + padding));
      const y = -70 - padding - (Math.floor(i / rowSize)* (size + padding));
      const conceptIconStyle = 'background-position: ' + x + 'px ' + y + 'px;';
      return conceptIconStyle;
    }
    if (layerIndex === 3) {
      const size = 35;
      const padding = 35;
      const rowSize = 16;
      const x = -padding - ((i % rowSize) * (size + padding));
      const y = -210 - padding - (Math.floor(i / rowSize)* (size + padding));
      const conceptIconStyle = 'background-position: ' + x + 'px ' + y + 'px;';
      return conceptIconStyle;
    }
    if (layerIndex === 4) {
      const size = 35;
      const padding = 35;
      const rowSize = 16;
      const x = -padding - ((i % rowSize) * (size + padding));
      const y = -490 - padding - (Math.floor(i / rowSize)* (size + padding));
      const conceptIconStyle = 'background-position: ' + x + 'px ' + y + 'px;';
      return conceptIconStyle;
    }
    if (layerIndex === 5) {
      const size = 35;
      const padding = 35;
      const rowSize = 16;
      const x = -padding - ((i % rowSize) * (size + padding));
      const y = -770 - padding - (Math.floor(i / rowSize)* (size + padding));
      const conceptIconStyle = 'background-position: ' + x + 'px ' + y + 'px;';
      return conceptIconStyle;
    }
    if (layerIndex === 6) {
      const size = 70;
      const padding = 35;
      const rowSize = 11;
      const x = -10 - padding - ((i % rowSize) * (size + padding));
      const y = -10 - 1050 - padding - (Math.floor(i / rowSize)* (size + padding));
      const conceptIconStyle = 'background-position: ' + x + 'px ' + y + 'px;';
      return conceptIconStyle;
    }
    if (layerIndex === 7) {
      const size = 70;
      const padding = 35;
      const rowSize = 11;
      const x = -10 -padding - ((i % rowSize) * (size + padding));
      const y = -10-1365 - padding - (Math.floor(i / rowSize)* (size + padding));
      const conceptIconStyle = 'background-position: ' + x + 'px ' + y + 'px;';
      return conceptIconStyle;
    }
    return null;
  }

  async function loadLayer(layerIndex, order, spriteOrder) {
    const layer = 'conv' + layerIndex;
    const container = $('#' + layer);
    const path = './data/' + layer + '/';
    const sprite = path + 'top_matches_avg_contrast.png?' + Date.now(); // added date to break cache because of flask problem...
    let centersData = await getCentersData(path + 'centers_data.txt?' + Date.now());
    let topMatchesData = await getTopMatchesData(path + 'top_matches_data.txt?' + Date.now());
    let iconSize = 40.6;
    let padding = 3.5;

    if (layerIndex === 1) {
      iconSize = 47.3;
      padding = 6;
    }
    const rowSize = 10;
    const displayRowSize = 4;
    const displayRowSizeMatches = 10;
    //-40.5px -2.5px

    const rows = order;

    let tots = Array(centersData[0].length).fill(0).map(v => 0);
    centersData.forEach(center => center.forEach((act, i) => tots[i] += parseFloat(act)));
    const avgs = tots.map(tot => tot / centersData.length);

    const conceptsContainer = container.find('.concepts');
    let count = -1;
    rows.forEach((row, rowIndex) => {
      conceptsContainer.append('<tr></tr>');
      row.forEach((i, colIndex) => {
        count += 1;
        const displayIndex = count;
        const row = conceptsContainer.find('tr:last');
        // get the position in the sprite
        const x = -padding - ((i % rowSize) * iconSize);
        const y = -padding - (Math.floor(i / rowSize) * iconSize);
        let style = 'background-image: url(\'' + sprite + '\');';
        style += 'background-position: ' + x + 'px ' + y + 'px;';

        const conceptIconStyle = getConceptIcon(layerIndex, spriteOrder ? spriteOrder[rowIndex][colIndex] : 0);
        const id = layer + '_button_' + displayIndex;
        const elStr = '<td><button id="' + id + '" class="conceptSpriteIcon" style="' + conceptIconStyle + '"></button></td>';
        row.append(elStr);

        $('#' + id).click(() => {
          currentMatchIndex = [(layerIndex - 2), displayIndex, 0];
          console.log(id, displayIndex, i);
          // Show top match avg image for visual concept
          container.find('.centerIcon')[0].style = style;
          conceptsContainer.find('.conceptSpriteIcon').removeClass('selected');
          conceptsContainer.find('.conceptSpriteIcon').removeClass('selected').eq(displayIndex).addClass('selected');

          // Show bar chart for average activation
          const data_f_0 = centersData[i].map((v, k) => {
            return { v: parseFloat(v), i: k };
          });

          // show top matches
          const matchesContainer = container.find('.topMatches');
          matchesContainer.empty();
          const sprite = path + 'top_matches_' + (i < 100 ? '0' : '') + (i < 10 ? '0' : '') + i + '_contrast.png?' + Date.now();
          for (let j = 0; j < topMatchesData[i].length; j += 1) {
            if ((j % displayRowSizeMatches) === 0) {
              matchesContainer.append('<tr></tr>');
            }
            const row = matchesContainer.find('tr:last');
            // get the position in the sprite
            const x = -padding - ((j % rowSize) * iconSize);
            const y = -padding - (Math.floor(j / rowSize) * iconSize);
            let style = 'background-image: url(\'' + sprite + '\');';
            style += 'background-position: ' + x + 'px ' + y + 'px;';
            const id = layer + '_button_' + displayIndex + '_' + j;
            const elStr = '<td><button id="' + id + '" class="exampleIcon filterIcon" style="' + style + '"></button></td>';
            row.append(elStr);

            $('#' + id).click(() => {
              currentMatchIndex = [(layerIndex - 1), displayIndex, j];
              // Show top match avg image for visual concept
              container.find('.selectedExampleIcon')[0].style = style;
              matchesContainer.find('.exampleIcon').removeClass('selected');
              matchesContainer.find('.exampleIcon').removeClass('selected').eq(j).addClass('selected');

              // Show bar chart for average activation
              // const bs = biases[layerIndex - 2];

              const data_f_1 = topMatchesData[i][j].map((v, i) => {
                return { v: [parseFloat(v), data_f_0[i].v], i };
              });
              // const data_f_1c = topMatchesData[i][j].map((v, i) => {
              //   return { v: [avgs[i]], i };
              // });
              container.find('.selectedExampleBarChart').empty();
              drawActivationGraph(container.find('.selectedExampleBarChart')[0], data_f_1);
              // drawActivationGraph(container.find('.selectedExampleBarChart')[0], data_f_1c);

              if (layerIndex > 3) {
                const data_f_1b = topMatchesData[i][j].map((v, i) => {
                  return { v: [parseFloat(v) - avgs[i], data_f_0[i].v - avgs[i]], i };
                });
                drawActivationGraph(container.find('.selectedExampleBarChart')[0], data_f_1b);
              }
            });
          }
          matchesContainer.find('button:first').click();
        });
      });
    });

    conceptsContainer.find('button.conceptSpriteIcon:first').click();
  }

  // let order1 = [Array(14).fill(0).map((v, i) => i)];
  const order1 = [[1, 2, 9, 6, 10, 7, 4, 5, 11, 13, 8, 12, 0, 3]];
  const spriteOrder1 = [[2, 3, 4, 5, 6, 7, 11, 12, 13, 8, 9, 10, 0, 1]];
  loadLayer(1, order1, spriteOrder1);

  // const order2 = [
  //   [12, 14, 13, 25, 20, 26], // verts and hors
  //   [30, 27, 29, 15, 17, 16], // diags
  //   [28, 19, 3, 6, 7, 10], // fields
  //   [21, 11, 8, 18, 24, 22, 23, 2], // corners
  //   [0, 4, 1, 7, 5] // ends
  // ];
  const order2 = [
    [
      13, 14, 12, 25, 20, 26, // verts and hors
      15, 17, 16, 30, 27, 29, // diags
      28, 19, 3, 6, 7, 10, // fields
    ],
    [
      21, 11, 8, 18, 24, 22, 23, 2, // corners
      0, 4, 1, 9, 5 // ends
    ],
  ];
  const spriteOrder2 = [
    [
      2, 3, 4, 5, 6, 7,  // verts and hors
      11, 12, 13, 8, 9, 10, // diags
      0, 1, 28, 29, 30, 15, // fields
    ],
    [
      16, 17, 18, 19, 20, 21, 22, 23, // corners
      24, 25, 26, 27, 14 // ends
    ],
  ];
  loadLayer(2, order2, spriteOrder2);

  let order3 = [
    [
      45, 51, 1, 36, 0, 2, // lines edges
      38, 47, 39, 42, 44, 41, // diags
      14, 13, 12, 50, // corners
      37, 43, 9, 35, // soft corners
    ],
    [
      48, 31, 32, 52, // little ends
      49, 19, 21, 24, // big ends
      34, 16, 46, 17, // shapes
      23, 3, 18, 20, // fields
    ],
    [
      5, 15, 4, 10, 28, // repeat lines, grass
      22, 33, 29, 25, 11, 26, 6, 27, 30, // pipes
    ]
  ];
  let spriteOrder3 = [
    [
      2, 3, 4, 5, 6, 7, // lines edges
      8, 10, 9, 11, 12, 13, // diags
      16, 17, 18, 19, // corners
      20, 21, 22, 23, // soft corners
    ],
    [
      24, 25, 26, 27, // little ends
      28, 29, 30, 31, // big ends
      32, 33, 34, 35, // shapes
      0, 3, 36, 37, // fields
    ],
    [
      14, 15, 40, 38, 39, // repeat lines, grass
      41, 43, 45, 42, 44, 46, 47, 48, 49, // pipes
    ]
  ];
  loadLayer(3, order3, spriteOrder3);

  let order4 = [
    [
      4, 7, 49, 15, 6, 24, // lines edges
      52, 46, 32, 44, 55, // corners
      5, 33, 22, 0, // rounded corners
      19, 34, 36, 40, 41, 21, 48, // objects
    ],
    [
      1, 18, 20, 47, // tips
      51, // field patterns
      2, 50, 12, 8, 37, 39, 28, 17, // edge patterns
      9, 23, 26, 10, 29, 30, // pipes
    ],
  ];
  let spriteOrder4 = [
    [
      0, 1, 3, 4, 5, 6, // lines edges
      9, 10, 7, 8, 11, // corners
      26, 27, 28, 29, // rounded corners
      12, 13, 14, 15, 16, 49, 52, // objects
    ],
    [
      36, 37, 34, 35, // tips
      53,  // field patterns
      38, 46, 40, 39, 44, 45, 43, 41, // edge patterns
      20, 21, 19, 18, 17, 51, // pipes
    ],
  ];
  // let spriteOrder4Remain = [ // BADDDDD
  //   [
  //     33, 28, 28, 28, 32, 27, 29, 27, // rounded corners
  //   ],
  //   [
  //     45, 54, // bulges
  //   ],
  //   [
  //     2, 2,
  //     44, 44,
  //   ]
  // ];

  // const order4remain = [Array(54).fill(0).map((v, i) => i)];
  // order4 = [order4remain[0].filter(v => !order4.flat().includes(v))];
  // console.log(order4.flat());
  loadLayer(4, order4, spriteOrder4);

  // let order5 = [];
  // let row = null;
  // for (let i = 0; i < 57; i += 1) {
  //   if (i % 10 === 0) {
  //     if (row) {
  //       order5.push(row);
  //     }
  //     // if (i === 0) {
  //     //   row = [19];
  //     // } else {
  //     //   row = [];
  //     // }
  //     row = [];
  //   }
  //   row.push(i);
  // }
  // order5.push(row);
  let order5 = [
    [
      39, 22, 37, 34, 31, // corners
      33, 35, 42, 5, // soft corners
      15, 16, 25, 15, // pipes
      54, 26, 14, // more pipes
      6, 17, 1, 43, // tips
    ],
    [
      19, 20, 52, 23, // edges
      7, 9, 13, 50, 49, 56, 53, 55, 51, // patterns
      8, 10, 11, 12, // objects
    ],
  ];
  let spriteOrder5 = [
    [
      14, 0, 1, 2, 3, // corners
      12, 13, 10, 11, // soft corners
      4, 5, 6, 7, // pipes
      29, 30, 24, // more pipes
      16, 35, 36, 37, // tips
    ],
    [
      25, 26, 44, 28, // edges
      17, 19, 23, 38, 42, 43, 45, 47, 48, // patterns
      18, 20, 21, 22, // objects
    ],
  ];

  // const order5remain = [Array(57).fill(0).map((v, i) => i)];
  // order5 = [order5remain[0].filter(v => !order5.flat().includes(v))];
  // console.log(order5.flat());
  loadLayer(5, order5, spriteOrder5);

  let order6 = [];
  let row6 = null;
  for (let i = 0; i < 25; i += 1) {
    if (i % 13 === 0) {
      if (row6) {
        order6.push(row6);
      }
      row6 = [];
    }
    row6.push(i);
  }
  order6.push(row6);
  let spriteOrder6 = [
    [
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    ],
    [
      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    ],
  ];
  loadLayer(6, order6, spriteOrder6);

  let order7 = [];
  let row7 = null;
  for (let i = 0; i < 25; i += 1) {
    if (i % 13 === 0) {
      if (row7) {
        order7.push(row7);
      }
      row7 = [];
    }
    row7.push(i);
  }
  order7.push(row7);
  let spriteOrder7 = [
    [
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    ],
    [
      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    ],
  ];
  loadLayer(7, order7, spriteOrder7);
}());
