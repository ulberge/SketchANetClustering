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
    e.preventDefault();
  };

  function drawActivationGraph(el, data) {
    const colors = ['#ff0000', '#0000ff'];
    const color = '#000';
    const margin = {top: 10, right: 10, bottom: 10, left: 10};
    const width = Math.min(data.length * 100, el.offsetWidth - margin.left - margin.right) + margin.left + margin.right;
    const height = width * 0.35;

    const x = d3.scaleBand()
      .domain(data.map(d => d.i))
      .range([margin.left, width - margin.right])
      .padding(0.1);

    const y = d3.scaleLinear()
      .domain(d3.extent(data, d => d.v[1] * 1.4)).nice()
      .range([height - margin.bottom, margin.top]);

    const xAxis = g => g
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).tickSizeOuter(0));

    // const yAxis = g => g
    //   .attr('transform', `translate(${margin.left},0)`)
    //   .call(d3.axisLeft(y)
    //     .ticks(6)
    //     .tickFormat(d => (d * 100) + '%')
    //   );

    const svg = d3.select(el).append('svg')
      .attr('width', width)
      .attr('height', height);

    const bars = svg.append('g').selectAll('.bar').data(data).enter();

    // bars.append('rect')
    //     .attr('fill', color)
    //     .attr('x', d => x(d.i))
    //     .attr('y', d => y(d.v))
    //     .attr('height', d => y(0) - y(d.v ? d.v : 0))
    //     .attr('width', x.bandwidth());
    const barCount = data[0].v.length;
    for (let i = 0; i < barCount; i++) {
      bars.append('rect')
        .attr('fill', colors[i])
        .attr('x', d => x(d.i) + (i * (x.bandwidth() / barCount)))
        .attr('y', d => y(d.v[i] ? d.v[i] : 0))
        .attr('height', d => y(0) - y(d.v[i] ? d.v[i] : 0))
        .attr('width', x.bandwidth() / barCount);
    }

    const xAxisItems = svg.append('g')
          .style('font-size', '8px')
          .call(xAxis);

    xAxisItems.selectAll('.tick').remove();

    // const yAxisItems = svg.append('g')
    //   .style('font-size', '10px')
    //   .call(yAxis);

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
        console.log(topMatchesData);
        resolve(topMatchesData);
      });
    });
  }

  async function loadLayer(layerIndex) {
    const layer = 'conv' + layerIndex;
    const container = $('#' + layer);
    const path = './data/' + layer + '/';
    const sprite = path + 'top_matches_avg.png?' + Date.now(); // added date to break cache because of flask problem...
    let centersData = await getCentersData(path + 'centers_data.txt?' + Date.now());
    let topMatchesData = await getTopMatchesData(path + 'top_matches_data.txt?' + Date.now());
    const iconSize = 28;
    const padding = 2;
    const rowSize = 10;
    const displayRowSize = 4;
    const displayRowSizeMatches = 10;

    const rows = [
      [12, 14, 13, 25, 20, 26], // verts and hors
      [30, 27, 29, 15, 17, 16], // diags
      [28, 19, 3, 6, 7, 10], // fields
      [21, 11, 8, 18, 24, 22, 23, 2], // corners
      [0, 4, 1, 7, 5] // ends
    ];

    const conceptsContainer = container.find('.concepts');
    let count = -1;
    rows.forEach(row => {
      conceptsContainer.append('<tr></tr>');
      row.forEach(i => {
        count += 1;
        const displayIndex = count;
        const row = conceptsContainer.find('tr:last');
        // get the position in the sprite
        const x = -padding - ((i % rowSize) * iconSize);
        const y = -padding - (Math.floor(i / rowSize) * iconSize);
        let style = 'background-image: url(\'' + sprite + '\');';
        style += 'background-position: ' + x + 'px ' + y + 'px;';
        const id = name + '_button_' + displayIndex;
        const elStr = '<td><button id="' + id + '" class="conceptIcon filterIcon" style="' + style + '"></button></td>';
        row.append(elStr);

        $('#' + id).click(() => {
          currentMatchIndex = [(layerIndex - 2), displayIndex, 0];
          console.log(id, displayIndex, i);
          // Show top match avg image for visual concept
          container.find('.centerIcon')[0].style = style;
          conceptsContainer.find('.conceptIcon').removeClass('selected');
          conceptsContainer.find('.conceptIcon').removeClass('selected').eq(displayIndex).addClass('selected');

          // Show bar chart for average activation
          const data_f_0 = centersData[i].map((v, k) => {
            return { v: parseFloat(v), i: k };
          });
          // container.find('.centerBarChart').empty();
          // drawActivationGraph(container.find('.centerBarChart')[0], data_f);

          // show top matches
          const matchesContainer = container.find('.topMatches');
          matchesContainer.empty();
          const sprite = path + 'top_matches_' + (i < 100 ? '0' : '') + (i < 10 ? '0' : '') + i + '.png?' + Date.now();
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
            const id = name + '_button_' + displayIndex + '_' + j;
            const elStr = '<td><button id="' + id + '" class="exampleIcon filterIcon" style="' + style + '"></button></td>';
            row.append(elStr);

            $('#' + id).click(() => {
              currentMatchIndex = [(layerIndex - 2), displayIndex, j];
              // Show top match avg image for visual concept
              container.find('.selectedExampleIcon')[0].style = style;
              matchesContainer.find('.exampleIcon').removeClass('selected');
              matchesContainer.find('.exampleIcon').removeClass('selected').eq(j).addClass('selected');

              // Show bar chart for average activation
              const data_f_1 = topMatchesData[i][j].map((v, i) => {
                return { v: [parseFloat(v), data_f_0[i].v], i };
              });
              container.find('.selectedExampleBarChart').empty();
              drawActivationGraph(container.find('.selectedExampleBarChart')[0], data_f_1);
            });
          }

          matchesContainer.find('button:first').click();
        });
      });
    });

    conceptsContainer.find('button.conceptIcon:first').click();
  }

  loadLayer(2);
}());
