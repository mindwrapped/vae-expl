// Displays the points from the latent space and dimensionally reduced points.
class PointChart {
    constructor(imageCallback, imagesCallback, valueCallback, interpolateCallback) {
        // Set default values
        this.svg = d3.select('#pointChart');
        this.latentDim = 2;
        this.data = [];
        this.points = [];
        this.density = 80;
        this.x = 0.0;
        this.y = 0.0;

        // Set callbacks
        this.imageCallback = imageCallback;
        this.imagesCallback = imagesCallback;
        this.valueCallback = valueCallback;
        this.interpolateCallback = interpolateCallback;

        // Make a points g and create a scale.
        this.pointsGroup = this.svg.append('g').attr('id', 'g-points');
        this.xScale = d3.scaleLinear()
            .domain([-1.0, 1.0])
            .range([30, 630]);
        this.yScale = d3.scaleLinear()
            .domain([-1.0, 1.0])
            .range([630, 30]);

        // Add axis groups.
        this.xAxisGroup = this.svg.append('g')
            .attr('id', 'g-x-axis');
        this.yAxisGroup = this.svg.append('g')
            .attr('id', 'g-y-axis');

        // Create a color scale for the labels.
        let labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
        this.colorScale = d3.scaleOrdinal()
            .domain(labels)
            .range(d3.schemeCategory10);

        // Set selected points to be undefined.
        this.selectedPoint1 = undefined;
        this.selectedPoint2 = undefined;
    }

    updateChart() {
        let that = this;

        // If there are points then change min and max values.
        if (this.points.length !== 0) {
            this.xMin = d3.min(that.points.map(d => d.point[0]));
            this.xMax = d3.max(that.points.map(d => d.point[0]));
            this.yMin = d3.min(that.points.map(d => d.point[1]));
            this.yMax = d3.max(that.points.map(d => d.point[1]));
        }

        // If there are not points then use range of -1 to 1.
        if (this.latentDim === 2 && this.points.length === 0) {
            this.xMin = -1.0;
            this.xMax = 1.0;
            this.yMin = -1.0;
            this.yMax = 1.0;
        }

        // Update scales based on mins and maxes
        this.xScale = d3.scaleLinear()
            .domain([that.xMin, that.xMax])
            .range([30, 630]);
        this.yScale = d3.scaleLinear()
            .domain([that.yMin, that.yMax])
            .range([630, 30]);

        // Create background boxes as needed for 2d latent dimension.
        this.data = [];
        if (this.latentDim === 2) {
            for (let i = this.xMin; i <= this.xMax; i += (this.xMax - this.xMin)/that.density) {
                for (let j = this.yMin; j <= this.yMax; j += (this.yMax - this.yMin)/that.density) {
                    this.data.push({
                        x: i,
                        y: j,
                    })
                }
            }
        }

        // Draw rects on graph.
        this.pointsGroup.selectAll('rect').data(that.data)
            .join(enter => enter.append("rect")
                .attr('fill', 'lightgrey')
                .attr('x', d => that.xScale(d.x))
                .attr('y', d => that.yScale(d.y))
                .attr('height', 5)
                .attr('width', 5)
                .attr('id', d => 'rect-' + d.x + '-' + d.y)
                .attr('class', 'point')
                .on("mouseover", function(d) {
                    that.updateLabels(d);
                    that.imageCallback(d);
                }),
            update => update
                .attr('fill', 'lightgrey')
                .attr('x', d => that.xScale(d.x))
                .attr('y', d => that.yScale(d.y))
                .attr('height', 5)
                .attr('width', 5)
                .attr('id', d => 'rect-' + d.x + '-' + d.y)
                .attr('class', 'point')
                .on("mouseover", function(d) {
                    that.updateLabels(d);
                    that.imageCallback(d);
                }),
            exit => exit.remove()
        );

        // Display x and y axis.
        let xAxis = d3.axisBottom().scale(that.xScale);
        this.xAxisGroup
            .call(xAxis)
            .attr('transform', 'translate(0,635)');
        let yAxis = d3.axisLeft().scale(that.yScale);
        this.yAxisGroup
            .call(yAxis)
            .attr('transform', 'translate(30,5)');

        // Display all data points.
        this.pointsGroup.selectAll('circle').data(that.points)
            .join(enter => enter.append("circle")
                    .attr('fill', function(d) {
                        return that.colorScale(d.label);
                    })
                    .attr('cx', d => that.xScale(d.point[0]))
                    .attr('cy', d => that.yScale(d.point[1]))
                    .attr('r', 3)
                    .attr('id', (d) => d.id)
                    .attr('class', 'point')
                    .on("mouseover", function(d) {
                        that.imagesCallback(d);
                        that.updateLabels(d);
                    })
                    .on("click", function(d) {
                        that.selectedPoint(d);
                    }),
                update => update.attr('fill', function(d) {
                        return that.colorScale(d.label);
                    })
                    .attr('cx', d => that.xScale(d.point[0]))
                    .attr('cy', d => that.yScale(d.point[1]))
                    .attr('r', 3)
                    .on("mouseover", function(d) {
                        that.imagesCallback(d);
                        that.updateLabels(d);
                    })
                    .on("click", function(d) {
                        that.selectedPoint(d);
                    }),
                exit => exit.remove()
            );

        // If selectedPoints are not undefined then make sure they are selected.
        if (this.selectedPoint1 !== undefined) {
            d3.select('#' + this.selectedPoint1.id).classed('selected1', true);
        }
        if (this.selectedPoint2 !== undefined) {
            d3.select('#' + this.selectedPoint2.id).classed('selected2', true);
        }
    }

    // Update point x and y labels.
    updateLabels(d) {
        if (d.point !== undefined) {
            this.valueCallback({x: d.point[0], y: d.point[1]});
        } else {
            this.valueCallback(d);
        }
    }

    // Update the displayed points.
    async updateDisplayedPoints(points) {
        this.points = points;
        this.updateChart();
    }

    // Update the latent dimension size.
    updateLatentDim(num) {
        this.latentDim = num;
        this.updateChart();
    }

    // Handle selected point click.
    selectedPoint(d) {
        let that = this;
        if (that.selectedPoint1 === undefined) {
            that.selectedPoint1 = d;

            d3.select('#' + d.id).classed('selected1', true);

            that.interpolateCallback(that.selectedPoint1, undefined);
        } else if (that.selectedPoint2 === undefined) {
            that.selectedPoint2 = d;

            that.interpolateCallback(that.selectedPoint1, that.selectedPoint2);

            d3.select('#' + d.id).classed('selected2', true);
        } else {
            d3.select('#' + that.selectedPoint1.id).classed('selected1', false);
            d3.select('#' + that.selectedPoint2.id).classed('selected2', false);

            that.selectedPoint1 = d;
            that.selectedPoint2 = undefined;

            that.interpolateCallback(that.selectedPoint1, undefined);

            d3.select('#' + d.id).classed('selected1', true);
        }
    }
}