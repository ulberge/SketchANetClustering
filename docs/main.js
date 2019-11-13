(function() {
  const layer = 'L1';
  const layerContainer = $('#' + layer);
  const neuronsContainer = layerContainer.find('.neurons');
  const topMatchesContainer = layerContainer.find('.topMatches');
  const neurons = [1, 2, 3, 4, 5, ];
  const rowSize = 5;
  const layerIndex = 0;

  function getVCIconStyle(i) {
    const rowSize = 10;
    const x = (-80 - ((i % rowSize) * (43 + 7.5))) / 1;
    const y = (-91 - (Math.floor(i / rowSize) * (42 + 88.5)));
    const style = 'background-image: url(\'./public/imgs/' + layer + '/top_matches_avg.png\'); background-position: ' + x + 'px ' + y + 'px;';
    return style;
  }

  function selectNeuron(id, i) {
    console.log('open ' + id);
    layerContainer.find('.neuronId').html(i);

    // Show larger version of icon and bar chart
    const iconStyle = getVCIconStyle(i);
    layerContainer.find('.centerIcon')[0].style = iconStyle;
    const barChartContainer = layerContainer.find('.centerBarChart');
    barChartContainer.empty();
    const barChartFile  = './public/imgs/' + layer + '/charts_0' + (i < 100 ? '0' : '') + (i < 10 ? '0' : '') + i + '.png';
    barChartContainer.append('<img src="' + barChartFile + '"/>')

    // Show top matches
    topMatchesContainer.empty();
    const imgFile  = 'top_matches_0' + (i < 100 ? '0' : '') + (i < 10 ? '0' : '') + i + '.png';
    const rowSize = 10;
    for (let i = 0; i < 100; i += 1) {
      if ((i % rowSize) === 0) {
        topMatchesContainer.append('<tr></tr>');
      }
      const row = topMatchesContainer.find('tr:last');
      const x = -3 + (i * 50);
      const y = -3 + (i * 50);
      const style = 'background-image: url("./public/imgs/' + layer + '/' + imgFile + '"); background-position: ' + x + 'px ' + y + 'px;';
      const ex_id = id + '_ex' + i;
      row.append('<td><button id="' + ex_id + '"><div class="filterIcon" style="' + style + '"></div></button></td>');

      topMatchesContainer.find('#' + ex_id).click(() => {
        console.log('open ' + ex_id);
      });
    }
  }

  for (let i = 0; i < 26; i += 1) {
    // add icon for this neuron
    if ((i % rowSize) === 0) {
      neuronsContainer.append('<tr></tr>');
    }
    const row = neuronsContainer.find('tr:last');
    const id = 'layer' + layerIndex + '_n' + i;
    const style = getVCIconStyle(i);

    row.append('<td><button id="' + id + '"><div class="filterIcon" style="' + style + '"></div></button></td>');

    neuronsContainer.find('#' + id).click(() => {
      selectNeuron(id, i);
    });
  }

  neuronsContainer.find('button:first').click();
  topMatchesContainer.find('button:first').click();
}());
