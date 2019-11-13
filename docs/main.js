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
    const color = '#000';
    const margin = {top: 10, right: 10, bottom: 10, left: 10};
    const width = Math.min(data.length * 100, el.offsetWidth - margin.left - margin.right) + margin.left + margin.right;
    const height = width * 0.35;

    const x = d3.scaleBand()
      .domain(data.map(d => d.i))
      .range([margin.left, width - margin.right])
      .padding(0.1);

    const y = d3.scaleLinear()
      .domain(d3.extent(data, d => d.v)).nice()
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

    bars.append('rect')
        .attr('fill', color)
        .attr('x', d => x(d.i))
        .attr('y', d => y(d.v))
        .attr('height', d => y(0) - y(d.v ? d.v : 0))
        .attr('width', x.bandwidth());

    const xAxisItems = svg.append('g')
          .style('font-size', '8px')
          .call(xAxis);

    xAxisItems.selectAll('.tick').remove();

    // const yAxisItems = svg.append('g')
    //   .style('font-size', '10px')
    //   .call(yAxis);

    return svg.node();
  }

  // function drawConcepts(layerContainer) {
  //   // draw all the concepts as icons for this layer as buttons in a grid
  //   const conceptsContainer = layerContainer.find('.concepts');
  //   drawSpriteAsButtonGrid(conceptsContainer, './public/imgs/conv2/top_matches_avg.png', 49, 3, 10, 60);
  // }

  // function drawSelectedConcept(layerContainer) {
  //   // draw a larger version of the selected concept's icon
  //   // use D3 to render its activations
  //   const data = [10, 20, 40, 4].map((v, i) => {
  //     return { v, i }
  //   });
  //   drawActivationGraph(layerContainer.find('.centerBarChart')[0], data);
  // }

  // function drawTopMatches() {
  //   // draw the top matches as icons in a grid
  // }

  // function drawSelectedTopMatch() {
  //   // draw the activations for the selected top match
  // }

  // drawConcepts($('#L1'));
  // drawSelectedConcept($('#L1'));

  // const layer = 'L1';
  // const layerContainer = $('#' + layer);
  // const neuronsContainer = layerContainer.find('.neurons');
  // const topMatchesContainer = layerContainer.find('.topMatches');
  // const neurons = [1, 2, 3, 4, 5, ];
  // const rowSize = 5;
  // const layerIndex = 0;

  // function getVCIconStyle(i) {
  //   const rowSize = 10;
  //   const x = (-3 - ((i % rowSize) * (49)));
  //   const y = (-3 - (Math.floor(i / rowSize) * (49)));
  //   const style = 'background-image: url(\'./data/conv2/top_matches_avg.png\'); background-position: ' + x + 'px ' + y + 'px;';
  //   return style;
  // }

  // function selectNeuron(id, i) {
  //   console.log('open ' + id);
  //   layerContainer.find('.neuronId').html(i);

  //   // Show larger version of icon and bar chart
  //   const iconStyle = getVCIconStyle(i);
  //   layerContainer.find('.centerIcon')[0].style = iconStyle;
  //   const barChartContainer = layerContainer.find('.centerBarChart');
  //   barChartContainer.empty();
  //   const barChartFile  = './public/imgs/' + layer + '/charts_0' + (i < 100 ? '0' : '') + (i < 10 ? '0' : '') + i + '.png';
  //   barChartContainer.append('<img src="' + barChartFile + '"/>')

  //   // Show top matches
  //   topMatchesContainer.empty();
  //   const imgFile  = 'top_matches_0' + (i < 100 ? '0' : '') + (i < 10 ? '0' : '') + i + '.png';
  //   const rowSize = 10;
  //   for (let i = 0; i < 100; i += 1) {
  //     if ((i % rowSize) === 0) {
  //       topMatchesContainer.append('<tr></tr>');
  //     }
  //     const row = topMatchesContainer.find('tr:last');
  //     const x = -3 + (i * 50);
  //     const y = -3 + (i * 50);
  //     const style = 'background-image: url("./public/imgs/' + layer + '/' + imgFile + '"); background-position: ' + x + 'px ' + y + 'px;';
  //     const ex_id = id + '_ex' + i;
  //     row.append('<td><button id="' + ex_id + '" class="filterIcon" style="' + style + '"></button></td>');

  //     topMatchesContainer.find('#' + ex_id).click(() => {
  //       console.log('open ' + ex_id);
  //     });
  //   }
  // }

  // for (let i = 0; i < 26; i += 1) {
  //   // add icon for this neuron
  //   if ((i % rowSize) === 0) {
  //     neuronsContainer.append('<tr></tr>');
  //   }
  //   const row = neuronsContainer.find('tr:last');
  //   const id = 'layer' + layerIndex + '_n' + i;
  //   const style = getVCIconStyle(i);

  //   row.append('<td><button id="' + id + '" class="filterIcon" style="' + style + '"></button></td>');

  //   neuronsContainer.find('#' + id).click(() => {
  //     selectNeuron(id, i);
  //   });
  // }

  // neuronsContainer.find('button:first').click();
  // topMatchesContainer.find('button:first').click();

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
        const topMatchesData = data.split(':').map(g => g.split('\n').map(v => v.split(',')));
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
    const centersData = await getCentersData(path + 'centers_data.txt');
    const topMatchesData = await getTopMatchesData(path + 'top_matches_data.txt');
    const iconSize = 28;
    const padding = 2;
    const rowSize = 10;
    const numIcons = 4;

    const conceptsContainer = container.find('.concepts');
    for (let i = 0; i < numIcons; i += 1) {
      if ((i % rowSize) === 0) {
        conceptsContainer.append('<tr></tr>');
      }
      const row = conceptsContainer.find('tr:last');
      // get the position in the sprite
      const x = -padding - ((i % rowSize) * iconSize);
      const y = -padding - (Math.floor(i / rowSize) * iconSize);
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
        const data_f = centersData[i].map((v, i) => {
          return { v: parseFloat(v), i };
        });
        container.find('.centerBarChart').empty();
        console.log(data_f);
        drawActivationGraph(container.find('.centerBarChart')[0], data_f);

        // show top matches
        const matchesContainer = container.find('.topMatches');
        matchesContainer.empty();
        const sprite = path + 'top_matches_' + (i < 100 ? '0' : '') + (i < 10 ? '0' : '') + i + '.png';
        for (let j = 0; j < 100; j += 1) {
          if ((j % rowSize) === 0) {
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
            conceptsContainer.find('.exampleIcon').removeClass('selected');
            conceptsContainer.find('.exampleIcon').removeClass('selected').eq(i).addClass('selected');

            // Show bar chart for average activation
            const data_f = topMatchesData[i][j].map((v, i) => {
              return { v: parseFloat(v), i };
            });
            container.find('.selectedExampleBarChart').empty();
            drawActivationGraph(container.find('.selectedExampleBarChart')[0], data_f);
          });
        }

        matchesContainer.find('button:first').click();
      });
    }

    conceptsContainer.find('button:first').click();

  }

  loadLayer(2);
}());
