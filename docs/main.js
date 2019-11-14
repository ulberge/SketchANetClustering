(function() {

  function drawSpriteAsButtonGrid(name, container, sprite, iconSize, padding, rowSize, numIcons) {
  }

  function selectConcept(container, name, id, style) {
    // Show larger version of icon and bar chart
    container.find('.centerIcon')[0].style = style;
    // const barChartContainer = layerContainer.find('.centerBarChart');
    // barChartContainer.empty();
    // const barChartFile  = './public/imgs/' + layer + '/charts_0' + (i < 100 ? '0' : '') + (i < 10 ? '0' : '') + i + '.png';
    // barChartContainer.append('<img src="' + barChartFile + '"/>')

    // // Show top matches
    // topMatchesContainer.empty();
    // const imgFile  = 'top_matches_0' + (i < 100 ? '0' : '') + (i < 10 ? '0' : '') + i + '.png';
    // const rowSize = 10;
    // for (let i = 0; i < 100; i += 1) {
    //   if ((i % rowSize) === 0) {
    //     topMatchesContainer.append('<tr></tr>');
    //   }
    //   const row = topMatchesContainer.find('tr:last');
    //   const x = -3 + (i * 50);
    //   const y = -3 + (i * 50);
    //   const style = 'background-image: url("./public/imgs/' + layer + '/' + imgFile + '"); background-position: ' + x + 'px ' + y + 'px;';
    //   const ex_id = id + '_ex' + i;
    //   row.append('<td><button id="' + ex_id + '" class="filterIcon" style="' + style + '"></button></td>');

    //   topMatchesContainer.find('#' + ex_id).click(() => {
    //     console.log('open ' + ex_id);
    //   });
    // }
  }

  // function drawActivationGraph(el, data) {
  //   var margin = {top: 40, right: 20, bottom: 30, left: 40},
  //       width = 960 - margin.left - margin.right,
  //       height = 500 - margin.top - margin.bottom;

  //   var formatPercent = d3.format(".0%");

  //   var x = d3.scaleBand()
  //       .range([0, width], .1)
  //       .padding(0.1);;

  //   var y = d3.scaleLinear()
  //       .range([height, 0]);

  //   var xAxis = d3.svg.axis()
  //       .scale(x)
  //       .orient("bottom");

  //   var yAxis = d3.svg.axis()
  //       .scale(y)
  //       .orient("left")
  //       .tickFormat(formatPercent);

  //   var tip = d3.tip()
  //     .attr('class', 'd3-tip')
  //     .offset([-10, 0])
  //     .html(function(d) {
  //       return "<strong>Frequency:</strong> <span style='color:red'>" + d.frequency + "</span>";
  //     });

  //   var svg = d3.select("body").append("svg")
  //       .attr("width", width + margin.left + margin.right)
  //       .attr("height", height + margin.top + margin.bottom)
  //     .append("g")
  //       .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  //   svg.call(tip);

  //   x.domain(data.map(function(d) { return d.letter; }));
  //   y.domain([0, d3.max(data, function(d) { return d.frequency; })]);

  //   svg.append("g")
  //       .attr("class", "x axis")
  //       .attr("transform", "translate(0," + height + ")")
  //       .call(xAxis);

  //   svg.append("g")
  //       .attr("class", "y axis")
  //       .call(yAxis)
  //     .append("text")
  //       .attr("transform", "rotate(-90)")
  //       .attr("y", 6)
  //       .attr("dy", ".71em")
  //       .style("text-anchor", "end")
  //       .text("Frequency");

  //   svg.selectAll(".bar")
  //       .data(data)
  //     .enter().append("rect")
  //       .attr("class", "bar")
  //       .attr("x", function(d) { return x(d.letter); })
  //       .attr("width", x.rangeBand())
  //       .attr("y", function(d) { return y(d.frequency); })
  //       .attr("height", function(d) { return height - y(d.frequency); })
  //       .on('mouseover', tip.show)
  //       .on('mouseout', tip.hide);
  // }

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
      .domain(d3.extent(data, d => Math.max(d.v[0], d.v[1]))).nice()
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
        const centersData = data.split('\n').map(v => v.split(','));
        centersData.pop();
        resolve(centersData);
      });
    });
  }

  async function getTopMatchesData(fileName) {
    return new Promise((resolve) => {
      $.get(fileName, function(data) {
        const topMatchesData = data.split(':').map(g => g.trim().split('\n').map(v => v.split(',')));
        console.log(topMatchesData);
        topMatchesData.pop();
        resolve(topMatchesData);
      });
    });
  }

  async function loadLayer(i) {
    const layer = 'conv' + i;
    const container = $('#' + layer);
    const path = './data/' + layer + '/';
    const sprite = path + 'top_matches_avg.png';
    let centersData = await getCentersData(path + 'centers_data.txt');
    let topMatchesData = await getTopMatchesData(path + 'top_matches_data.txt');
    const iconSize = 28;
    const padding = 2;
    const rowSize = 10;
    const displayRowSize = 4;
    const displayRowSizeMatches = 10;

    const sortIdx = [
      2, 1, 15, 5,
      14, 10, 12, 8,
      16, 13, 11, 9,
      6, 4, 3, 0,
      7, 17
    ];
    centersData = centersData.map((d, i) => [sortIdx.indexOf(i), d]).sort((a, b) => a[0] > b[0] ? -1 : 1).map(d => d[1]);
    topMatchesData = topMatchesData.map((d, i) => [i, d]).sort((a, b) => sortIdx[a[0]] > sortIdx[b[0]] ? -1 : 1).map(d => d[1]);

    const conceptsContainer = container.find('.concepts');
    // for (let k = 0; k < sortIdx.length; k += 1) {
    //   const i = sortIdx[k];
    for (let i = 0; i < centersData.length; i += 1) {
      if ((i % displayRowSize) === 0) {
        conceptsContainer.append('<tr></tr>');
      }
      const row = conceptsContainer.find('tr:last');
      // get the position in the sprite
      const x = -padding - ((sortIdx[i] % rowSize) * iconSize);
      const y = -padding - (Math.floor(sortIdx[i] / rowSize) * iconSize);
      let style = 'background-image: url(\'' + sprite + '\');';
      style += 'background-position: ' + x + 'px ' + y + 'px;';
      const id = name + '_button_' + i;
      const elStr = '<td><button id="' + id + '" class="conceptIcon filterIcon" style="' + style + '"></button></td>';
      row.append(elStr);

      $('#' + id).click(() => {
        // Show top match avg image for visual concept
        container.find('.centerIcon')[0].style = style;
        conceptsContainer.find('.conceptIcon').removeClass('selected');
        conceptsContainer.find('.conceptIcon').removeClass('selected').eq(i).addClass('selected');

        // Show bar chart for average activation
        const data_f_0 = centersData[i].map((v, i) => {
          return { v: parseFloat(v), i };
        });
        // container.find('.centerBarChart').empty();
        // drawActivationGraph(container.find('.centerBarChart')[0], data_f);

        // show top matches
        const matchesContainer = container.find('.topMatches');
        matchesContainer.empty();
        const sprite = path + 'top_matches_' + (i < 100 ? '0' : '') + (i < 10 ? '0' : '') + i + '.png';
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
          const id = name + '_button_' + i + '_' + j;
          const elStr = '<td><button id="' + id + '" class="exampleIcon filterIcon" style="' + style + '"></button></td>';
          row.append(elStr);

          $('#' + id).click(() => {
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
    }

    conceptsContainer.find('button:first').click();

  }

  loadLayer(2);
}());
