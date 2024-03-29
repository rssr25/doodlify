
var contentImageName = ""
function readURL(input) {
  if (input.files && input.files[0]) {

    var reader = new FileReader();
    contentImageName = input.files[0].name

    reader.onload = function(e) {
      $('.image-upload-wrap').hide();

      $('.file-upload-image').attr('src', e.target.result);
      $('.file-upload-content').show();

      $('.image-title').html(input.files[0].name);
      document.getElementById('contentImageName').value = contentImageName;
    };

    reader.readAsDataURL(input.files[0]);

  } else {
    removeUpload();
  }
}

function removeUpload() {
  $('.file-upload-input').replaceWith($('.file-upload-input').clone());
  $('.file-upload-content').hide();
  $('.image-upload-wrap').show();
}

function imageOnClick(event)
{
	
	console.log(event.currentTarget.id);
	var images = ['image1', 'image2', 'image3']
	for(i = 0; i < images.length; i++)
	{
		if(event.currentTarget.id == images[i])
		{
			if(images[i] == "image1")
			{
				document.getElementById(images[i]).className = 'imageClicked'
				document.getElementById('styleImageName').value = images[i] + '.jpg'
			}
			else
			{
				document.getElementById(images[i]).className = 'imageClicked'
				document.getElementById('styleImageName').value = images[i] + '.png'
			}
			
		}
		else
		{
			document.getElementById(images[i]).className = 'image'
		}
	}

}

function buttonOnclick()
{
	console.log("button clicked")
	document.getElementById('loaderAnimation').style.opacity = "1.0";

}

$('.image-upload-wrap').bind('dragover', function () {
		$('.image-upload-wrap').addClass('image-dropping');
	});
	$('.image-upload-wrap').bind('dragleave', function () {
		$('.image-upload-wrap').removeClass('image-dropping');
});
