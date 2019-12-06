(function() {

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
