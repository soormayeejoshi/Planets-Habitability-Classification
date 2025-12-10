function newWindow(url, name, w, h) {
   var wstr = '';
   var hstr = '';
   if (w)
   {
     if (w == 'W_XLARGE') {wstr='width=1200,';}
     else if (w == 'W_LARGE') {wstr='width=800,';}
     else if (w == 'W_MEDIUM') {wstr='width=400,';}
     else if (w == 'W_SMALL') {wstr='width=250,';}
     else {wstr = 'width=' + w + ',';}
   }
   if (h)
   {
     if (h == 'H_XLARGE') {hstr = 'height=1200,';}
     else if (h == 'H_LARGE') {hstr = 'height=800,';}
     else if (h == 'H_MEDIUM') {hstr = 'height=400,';}
     else if (h == 'H_SMALL') {hstr = 'height=250,';}
     else {hstr = 'height=' + h + ','; }
   } 
   var str = wstr + hstr + 'scrollbars,resizable';
   var my_window=window.open(url,name,str);
   if (window.focus) {my_window.focus()}
   return false;
}

function newWindowLoc(url, name, w, h) {
   var wstr = '';
   var hstr = '';
   if (w)
   {
     if (w == 'W_XLARGE') {wstr='width=1200,';}
     else if (w == 'W_LARGE') {wstr='width=800,';}
     else if (w == 'W_MEDIUM') {wstr='width=400,';}
     else {wstr = 'width=' + w + ',';}
   }
   if (h)
   {
     if (h == 'H_XLARGE') {hstr = 'height=1200,';}
     else if (h == 'H_LARGE') {hstr = 'height=800,';}
     else if (h == 'H_MEDIUM') {hstr = 'height=400,';}
     else if (h == 'H_SMALL') {hstr = 'height=250,';}
     else {hstr = 'height=' + h + ',';}
   } 
   var str = wstr + hstr + "location,scrollbars,resizable,menubar,toolbar,directories,status";
   var my_window=window.open(url,name,str);
   if (window.focus) {my_window.focus()}
   return false;
}

function message(inMsg) {
  var the_name=confirm(inMsg);
  if (the_name) 
      return true;
  else
      return false;
}


function objMsg(obj, source) {
  var the_name=confirm("The "+ source + " object: "+obj+ " has been resolved. Would you like to continue?");
  if (the_name) {
    alert("yes");
    return true;
  } else {
    document.forms['the_form'].objmsg.value = "no"; 
    alert("no");
    return false;
  }
}


function sectionCheck(name) {

    var checkboxs = window.document.forms['the_form'].getElementsByTagName('input'); 

    for (var i=0, num; num=checkboxs[i]; ++i) {
        if (num.type == 'checkbox' && 
	    num.className == name  &&
	    num.checked == false)
	    num.checked = true;
    }
}

function sectionUncheck(name) {

    var checkboxs = window.document.forms['the_form'].getElementsByTagName('input'); 

    for (var i=0, num; num=checkboxs[i]; ++i) {

        if (num.type == 'checkbox' && 
	    num.className == name  &&
	    num.checked == true)
	    num.checked = false;
    }
}

function clearTextField(name) {

    var texts = window.document.forms['the_form'].getElementsByTagName('input'); 
    for (var i=0; i<texts.length; ++i) {

        if ((texts[i].type == 'text' || texts[i].type == 'hidden')  && texts[i].className == name) {
	    texts[i].value = ' ';
        }
    }
}

function ignore() {
}

function externalLinks() {
  if (!document.getElementsByTagName) return;
  var anchors = document.getElementsByTagName("a");
  for (var i = 0; i < anchors.length; i++) 
  {
    var anchor = anchors[i];
    if (anchor.getAttribute("href") && 
        anchor.getAttribute("rel"))
    {
      anchor.target = anchor.getAttribute("rel");
    }
  }
}

function checkbyname(checkValue, nametocheck)
{
	var endStatus=false;
	if (checkValue)
	{
		endStatus=true;
	}
	var allTargets = document.getElementsByName(nametocheck);
	
	for (var i = 0; i < allTargets.length; i++)
	{
		document.getElementById (allTargets[i].id).checked=endStatus;
	}
}

function checkParameters(checkValue, controlPrefix, controlSuffix)
{
	var endStatus=false;
	if (checkValue)
	{
		endStatus=true;
	}
	
	var SuffixLength = controlSuffix.length;
	var PrefixLength = controlPrefix.length;
	for (var PrefixCntr = 0; PrefixCntr < PrefixLength; PrefixCntr++)
	{
		for (var SuffixCntr = 0; SuffixCntr < SuffixLength; SuffixCntr++)
		{
			var controlID = controlPrefix[PrefixCntr] + controlSuffix[SuffixCntr];
			elem = document.getElementById (controlID);

			if (elem != null)
			{
				elem.checked=endStatus;
			}
		}
	}
}

function clearParameters(controlSuffix)
{
	var SuffixLength = controlSuffix.length;
	var thePrefixes = new Array('min_','max_');
	var PrefixLength = thePrefixes.length;
	

	for (var PrefixCntr = 0; PrefixCntr < PrefixLength; PrefixCntr++)
	{
		for (var SuffixCntr = 0; SuffixCntr < SuffixLength; SuffixCntr++)
		{
			var controlID = thePrefixes[PrefixCntr] + controlSuffix[SuffixCntr];
            
            // prevent if object doesn't exist -benny
            var controlIDobject=document.getElementById (controlID);
            if(controlIDobject)
            {
                document.getElementById (controlID).value="";
            }
		}
	}
}


function clearStarID()
{
    document.getElementById ("etssdetail").value="";
}

function clearCoordinateSearch()
{
    document.getElementById ("spat_coordinate").value="";
    document.getElementById ("spat_search_size").value="";
}
